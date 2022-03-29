from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize


def predict(classifier, testdataloader, loss_fn):
    classifier.eval()
    total_correct = 0
    total_testset = 0
    total_loss = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        loss = loss_fn(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_loss += loss.item()
        total_testset += points.size()[0]
    # print("final accuracy {}".format(total_correct / float(total_testset)))
    return total_loss / float(total_testset), total_correct / float(total_testset)


if __name__ == "__main__":
    loss_stats = {'train': [], "test": []}
    acc_stats = {'train': [], "test": []}
    for epoch in range(opt.nepoch):
        scheduler.step()
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            train_epoch_loss += loss.item()
            train_epoch_acc += correct.item() / float(opt.batchSize)
            if (i + 1) % 25 == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                    epoch, i + 1, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            # if i % 10 == 0:
            #     j, data = next(enumerate(testdataloader, 0))
            #     points, target = data
            #     target = target[:, 0]
            #     points = points.transpose(2, 1)
            #     points, target = points.cuda(), target.cuda()
            #     classifier = classifier.eval()
            #     pred, _, _ = classifier(points)
            #     loss = F.nll_loss(pred, target)
            #     pred_choice = pred.data.max(1)[1]
            #     correct = pred_choice.eq(target.data).cpu().sum()
            #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
            #         epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
        train_epoch_loss_avg = train_epoch_loss / len(dataloader)
        train_epoch_acc_avg = train_epoch_acc / len(dataloader)
        loss_stats['train'].append(train_epoch_loss_avg)
        acc_stats['train'].append(train_epoch_acc_avg)
        test_loss, test_acc = predict(classifier, testdataloader, F.nll_loss)
        loss_stats['test'].append(test_loss)
        acc_stats['test'].append(test_acc)
        print('[epoch: %d] %s train_loss: %f train_accuracy: %f  %s test_loss: %f test_accuracy: %f' % (
            epoch, blue('train'), train_epoch_loss_avg, train_epoch_acc_avg, blue('test'), test_loss, test_acc))
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # Create dataframes
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    train_test_acc_df = pd.DataFrame.from_dict(acc_stats).reset_index().melt(
        id_vars=['index']).rename(columns={"index": "epochs"})
    train_test_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(
        id_vars=['index']).rename(columns={"index": "epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_test_acc_df, x="epochs", y="value",
                 hue="variable", ax=axes[0]).set_title('Train-Test Accuracy/Epoch')
    sns.lineplot(data=train_test_loss_df, x="epochs", y="value",
                 hue="variable", ax=axes[1]).set_title('Train-Test Loss/Epoch')
    plt.show()
    _, final_acc = predict(classifier, testdataloader, F.nll_loss)
    print(f'The final accuracy is {final_acc}')

    # total_correct = 0
    # total_testset = 0
    # for i, data in tqdm(enumerate(testdataloader, 0)):
    #     points, target = data
    #     target = target[:, 0]
    #     points = points.transpose(2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     pred, _, _ = classifier(points)
    #     pred_choice = pred.data.max(1)[1]
    #     correct = pred_choice.eq(target.data).cpu().sum()
    #     total_correct += correct.item()
    #     total_testset += points.size()[0]
    #
    # print("final accuracy {}".format(total_correct / float(total_testset)))
