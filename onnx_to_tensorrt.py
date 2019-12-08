#!/usr/bin/env python2

from __future__ import print_function
import common
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from matplotlib import pyplot as plt
#from yolov3_to_onnx import download_file
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys
import os
import shutil
sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()

COCO = False

if COCO:
    OUT = 255
else:
    OUT = 75

SIZE = 608

# BUILD = '-f32'
BUILD = '-f16'
# BUILD = '-none'
# BUILD = '-f16-ns'

PRINT_RESULTS = False


#############################################################
# Standard NVIDIA Given Functions
#############################################################

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    # print(bboxes, confidences, categories)
    if (bboxes) is not None and (confidences) is not None and (categories) is not None :
        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            draw.text((left, top - 12),
                    '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            if BUILD == '-f16' or BUILD == '-f16-ns':
                print("BUILDING f16")
                builder.fp16_mode = True
            if BUILD == '-f32':
                print("BUILDING f32")
            if BUILD == '-int8':
                print("BUILDING int8")
                builder.int8_mode = True
            if BUILD != '-none' and BUILD != '-f16-ns':
                print("BUILDING with STRICT CONSTRAINTS")
                builder.strict_type_constraints = True            

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                    onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        #print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    if COCO:
        onnx_file_path = os.path.join('./engine/onnx/', 'yolov3-' + str(SIZE) + '.onnx')
        engine_file_path = os.path.join('./engine/trt/', 'yolov3-' + str(SIZE) + BUILD + '.trt')
    else:    
        onnx_file_path = os.path.join('./engine/onnx/', 'yolov3-voc-' + str(SIZE) + '.onnx')
        engine_file_path = os.path.join('./engine/trt/', 'yolov3-voc-' + str(SIZE) + BUILD + '.trt')

    # onnx_file_path = "./engine/yolov3-608.onnx"    
    # engine_file_path = "./engine/yolov3-608-voc-f32.trt"
    
    # loop over images
    if COCO:
        test_images_file = './coco/5k.txt'  #for coco
    else:
        test_images_file = './VOC/data/dataset/voc_test.txt'  #for voc

    with open(test_images_file, 'r') as f:
        txt = f.readlines()
        test_images = [line.strip() for line in txt]

    timeRecSave = []

    input_resolution_yolov3_HW = (SIZE, SIZE)

    predicted_dir_path = './mAP/predicted'
    if os.path.exists(predicted_dir_path):
        shutil.rmtree(predicted_dir_path)
    os.mkdir(predicted_dir_path)

    # ground_truth_dirs_path = './mAP/ground-truth'
    # if os.path.exists(ground_truth_dir_path):
    #     shutil.rmtree(ground_truth_dir_path)
    # os.mkdir(ground_truth_dir_path)

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        for idx, input_image_path in enumerate(test_images):

            #print("image path = ", input_image_path)
            filename = os.path.split(input_image_path)[1]
            #print("filename = ",filename)

            # try:
            #     label_file = './coco/labels/val2014/' + os.path.splitext(filename)[0]+'.txt'
            #     with open(label_file, 'r') as f:
            #         labels = f.readlines()
            # except:
            #     continue

            # Create a pre-processor object by specifying the required input resolution for YOLOv3
            preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
            # Load an image from the specified input path, and return it together with  a pre-processed version
            image_raw, image = preprocessor.process(input_image_path)
            # Store the shape of the original input image in WH format, we will need it for later
            # print("image shape = ", image.shape)
            # print("image data = ")
            # print(image) 
            shape_orig_WH = image_raw.size
            # print("image_raw.size = ", image_raw.size)
            # print("image_raw.shape = ", image_raw.shape)


            # Output shapes expected by the post-processor
            # output_shapes = [(1, 255, 10, 10), (1, 255, 20, 20), (1, 255, 40, 40)] #for 320
            # output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)] #for 416
            output_shapes = [(1, int(OUT), int(SIZE/32), int(SIZE/32)), (1, int(OUT), int(SIZE/16), int(SIZE/16)), (1, int(OUT), int(SIZE/8), int(SIZE/8))] #for 608

            # Do inference with TensorRT
            trt_outputs = []
            # with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
            #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
                # Do inference
                # print('Running inference on image {}...'.format(input_image_path)) # if idx==0 else 0
                # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = image
                # start = time.time()
            trt_outputs, timeRec = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                # print("time: %.2f s" %(time.time()-start))
                # print(trt_outputs)
            timeRecSave.append(timeRec)
            print('%d, Image %s, Recognition Time %0.3f seconds' % (idx, filename, timeRec))

            # # Before the post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

            # A list of 3 three-dimensional tuples for the YOLO masks
            # A list of 9 two-dimensional tuples for the YOLO anchors
            postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],   \
                                 "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), \
                                                (59, 119), (116, 90), (156, 198), (373, 326)],\
                                # Threshold for object coverage, float value between 0 and 1
                                "obj_threshold": 0.6,\
                                 # Threshold for non-max suppression algorithm, float value between 0 and 1
                                 "nms_threshold": 0.5,\
                                 "yolo_input_resolution": input_resolution_yolov3_HW}

            postprocessor = PostprocessYOLO(**postprocessor_args)

            # # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

             # Draw the bounding boxes onto the original input image and save it as a PNG file
            if PRINT_RESULTS:
                obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
                output_image_path = './results/yolo_' + filename
                obj_detected_img.save(output_image_path)
                print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

            predict_result_path = os.path.join(predicted_dir_path, str(idx) + '.txt')
            # ground_truth_path = os.path.join(ground_truth_dir_path, str(idx) + '.txt')

            with open(predict_result_path, 'w') as f:
                if boxes is not None:
                    for box, score, category_idx in zip(boxes, scores, classes):
                        x_coord, y_coord, width, height = box
                        box = [x_coord,y_coord, x_coord + width , y_coord + height] # fit YunYang1994's mAP calculation input format
                        category = ALL_CATEGORIES[category_idx]
                        category = "".join(category.split())
                        # print("score info = ", score, score.type)
                        box = list(map(int, box))
                        xmin, ymin, xmax, ymax = list(map(str, box))
                        # bbox_mess = ' '.join([category, score, xmin, ymin, xmax, ymax]) + '\n'
                        bbox_mess = ' '.join([category, "{:.4f}".format(score), xmin, ymin, xmax, ymax]) + '\n'
                        # print(bbox_mess)
                        f.write(bbox_mess)

    timeRecMean = np.mean(timeRecSave)
    print('The mean recognition time is {0:0.3f} seconds'.format(timeRecMean))

    # %%    Visualization of results
    if PRINT_RESULTS:
        np.save('results/timeRecognition.npy', timeRecSave)
        plt.figure(figsize=(8, 5))
        plt.plot(timeRecSave, label='Recg_time')
        plt.ylim([0, 0.05])
        plt.xlabel('Test image number'),
        plt.ylabel('Time [second]'),
        plt.title('Recognition time of Yolov3_DarkNet_ONNX_TensorRT_GPU_coco_test_2017')
        plt.hlines(y=timeRecMean, xmin=0, xmax=len(test_images),
                   linewidth=3, color='r', label='Mean')
        plt.savefig(
            'results/Yolov3_DarkNet_ONNX_TensorRT_GPU_coco_test_2017.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
