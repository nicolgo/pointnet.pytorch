cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(point_net LANGUAGES CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "The runtime output dirtory is: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "The library output dirtory is: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

add_library(render_balls_so SHARED render_balls_so.cpp)
