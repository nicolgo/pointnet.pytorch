# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

The project is forked from point.pytorch(https://github.com/fxia22/pointnet.pytorch.git)

# Download data and running

```
git clone https://github.com/nicolgo/pointnet.pytorch.git
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool (Linux)
```
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```
Build dynamic library on Windows
```
cd utils
mkdir build & cd build
cmake -DCMAKE_GENERATOR_PLATFORM=x64 .. or cmake -A x64 ..
cmake --build .

```
Training(for modelnet40, `dataset path` is useless since we rewrite the code getting modelnet40 dataset)
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

change the network with three symmetric functions(max/min/mean), just disable the comment of the code below in `model.py`
```
 # self.use_more_features = True
        if self.use_more_features:
```

# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | 89.2 | 
| this implementation(w/o feature transform) | 89.99 | 
| this implementation(w/ feature transform) | 89.59 | 

![image](https://user-images.githubusercontent.com/17155788/161382082-e988fa43-562b-41a3-851a-4835f9d270af.png)

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | N/A | 
| this implementation(w/o feature transform) | 97.59 | 
| this implementation(w/ feature transform) | 96.97 | 

![image](https://user-images.githubusercontent.com/17155788/161382101-351fb01c-de84-4e9b-af90-9f60960f9b65.png)

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:

![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)
