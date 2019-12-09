# CS230 Final Project

This repository was built from sample code provided by NVIDIA in the tensorrt package (/usr/src/tensorrt/samples/python/yolov3_onnx). This base code supports compiling YoloV3-608 from darkent weights to run a sample inference on a dog picture using COCO classes. Additionally, it includes mAP calculation SW from [here](https://github.com/YunYang1994/tensorflow-yolov3/tree/master/mAP) and the [coco dataset api](https://github.com/cocodataset/cocoapi)

The data_processing.py, yolov3_to_onnx.py, and onnx_to_tensorrt.py files were modified to:
- Build YoloV3-416 and YoloV3-320 as well
- Build YoloV3 models trained per VOC dataset requirements (20 classes)
- Build TensorRT models with Loop Fusion or Loop Fusion + f16 Quantization
- Convert the boundary boxes from TensorRT inference output into format able to be processed by main.py in /mAP.

Additional Work
- Compile ground truth labels for COCO and VOC data sets from source.

To use this repo:
1) Download the COCO val dataset use darkent [download scripts](https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh) as reference
2) Download the VOC dataset

## Object Detection With The ONNX TensorRT Backend In Python

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, yolov3_onnx, implements a full ONNX-based pipeline for performing inference with the YOLOv3 network, with an input size of 608 x 608 pixels, including pre and post-processing. This sample is based on the [YOLOv3-608](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.

## How does this sample work?

First, the original YOLOv3 specification from the paper is converted to the Open Neural Network Exchange (ONNX) format in `yolov3_to_onnx.py` (only has to be done once).

Second, this ONNX representation of YOLOv3 is used to build a TensorRT engine, followed by inference on a sample image in `onnx_to_tensorrt.py`. The predicted bounding boxes are finally drawn to the original input image and saved to disk.

After inference, post-processing including bounding-box clustering is applied. The resulting bounding boxes are eventually drawn to a new image file and stored on disk for inspection.

**Note:** This sample is not supported on Ubuntu 14.04 and older. Additionally, the `yolov3_to_onnx.py` script does not support Python 3.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1.  Install [ONNX-TensorRT: TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt). ONNX-TensorRT includes layer implementations for the required ONNX operators `Upsample` and `LeakyReLU`.

2.  Install the dependencies for Python.
	-   For Python 2 users, from the root directory, run:
	`python2 -m pip install -r requirements.txt`

	-   For Python 3 users, from the root directory, run:
	`python3 -m pip install -r requirements.txt`

## Running the sample

1.  Create an ONNX version of YOLOv3 with the following command. The Python script will also download all necessary files from the official mirrors (only once).
	`python yolov3_to_onnx.py`

	When running the above command for the first time, the output should look similar to the following:
	```
	Downloading from https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg, this may take a while...
	100% [................................................................................] 8342 / 8342
	Downloading from https://pjreddie.com/media/files/yolov3.weights, this may take a while...
	100% [................................................................................] 248007048 / 248007048
	[...]
	%106_convolutional = Conv[auto_pad = u'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]]
	(%105_convolutional_lrelu, %106_convolutional_conv_weights, %106_convolutional_conv_bias)
	return %082_convolutional, %094_convolutional,%106_convolutional
	}
	```

2.  Build a TensorRT engine from the generated ONNX file and run inference on a sample image, which will also be downloaded during the first run.
	`python onnx_to_tensorrt.py`

	When running the above command for the first time, the output should look similar to the following:
	```
	Downloading from https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg, this may take a while...
	100% [................................................................................] 163759 / 163759
	Building an engine from file yolov3.onnx, this may take a while...
	Running inference on image dog.jpg...
	Saved image with bounding boxes of detected objects to dog_bboxes.jpg.
	```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	Downloading from https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg, this may take a while…
	100% [......................................................................] 163759 / 163759
	Loading ONNX file from path yolov3.onnx...
	Beginning ONNX file parsing
	Completed parsing of ONNX file
	Building an engine from file yolov3.onnx; this may take a while...
	Completed creating Engine
	Running inference on image dog.jpg...
	[[135.14841333 219.59879284 184.30209195 324.0265199 ]
	  [ 98.30805074 135.72613533 499.71263299 299.25579652]
	  [478.00605802 81.25702449 210.57787895 86.91502688]] [0.99854713 0.99880403 0.93829258] [16 1 7]
	Saved image with bounding boxes of detected objects to dog_bboxes.png.
	```
	You should be able to visually confirm whether the detection was correct.

# Additional resources

The following resources provide a deeper understanding about the model used in this sample, as well as the dataset it was trained on:

**Model**
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

**Dataset**
- [COCO dataset](http://cocodataset.org/#home)

**Documentation**
- [YOLOv3-608 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
