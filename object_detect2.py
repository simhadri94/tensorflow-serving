import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import collections
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from tensorflow_serving.apis import predict_pb2
##from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import requests
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from PIL import Image
##from grpc.beta import implementations
import grpc

from utils import label_map_util
#from utils import visualization_utils as vis_util

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

#########################################################
#ip to connect to grpc server
##server = '192.168.1.9:8500' #'0.0.0.0:8500'
##host, port = server.split(':')

tf.app.flags.DEFINE_string('server', '192.168.1.9:8500','PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

# create the RPC stub
##channel = implementations.insecure_channel(host, int(port))
##stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

channel = grpc.insecure_channel('101.53.136.132:8500') #'0.0.0.0:8500')  # ('192.168.1.9:8500') ##FLAGS.server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# create the request object and set the name and signature_name params
request = predict_pb2.PredictRequest()
request.model_spec.name = 'ssd_inception_v2_coco' #'ssd_mobilenet_v2_coco'
request.model_spec.signature_name = 'serving_default'
#########################################################

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


#Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image):
    output = {}
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image))
    # sync requests
    output_dict = stub.Predict(request, 10.)
    #print(np.array(output_dict.outputs['detection_boxes'].float_val))
    
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output['num_detections'] = (np.array(output_dict.outputs['num_detections'].float_val)).astype(int)
    output['detection_classes'] = (np.array(output_dict.outputs['detection_classes'].float_val)).astype(np.uint8)
    output['detection_boxes'] = np.array(output_dict.outputs['detection_boxes'].float_val)
    output['detection_scores'] = np.array(output_dict.outputs['detection_scores'].float_val)
    
    #print("the result_future value is {}".format(result_future))
    #print("outputs['detection_classes']: {}".format(result_future.outputs['detection_classes']))
    
    return output

def draw_bounding_box(coordinates, image_array):
    x1, y1, x2, y2 = coordinates
    cv2.rectangle(image_array, (x1, y1), (x2, y2), (0,255,0),2)


def draw_text(coordinates, image_array, text, x_offset=0, y_offset=0,
                                                font_scale=0.8, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,255,0), thickness, cv2.LINE_AA)

######################################################################

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0) #'rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov')
while True:
    bgr_image = video_capture.read()[1]
    # gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # image_np = load_image_into_numpy_array(rgb_image)
    image_np_expanded = np.expand_dims(rgb_image, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded)

    output_dict['detection_boxes'] = output_dict['detection_boxes'].reshape((100,4))

    max_boxes_to_draw = 20
    min_score_thresh=.5
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    classes = output_dict['detection_classes']
    groundtruth_box_visualization_color='black'
    instance_masks=output_dict.get('detection_masks')
    use_normalized_coordinates=True
    line_thickness=8

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                display_str = str(class_name)
                if not display_str:
                    display_str = '{}%'.format(int(100*scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                
                box_to_color_map[box] = STANDARD_COLORS[
                    classes[i] % len(STANDARD_COLORS)]

    # # Draw all boxes onto image.
    # for box, color in box_to_color_map.items():
    #     ymin, xmin, ymax, xmax = box
    #     im_height, im_width = rgb_image.shape[:2]
    #     if use_normalized_coordinates:
    #         (x1, x2, y1, y2) = (int(xmin * im_width), int(xmax * im_width),
    #                               int(ymin * im_height), int(ymax * im_height))

    #         draw_bounding_box((x1, y1, x2, y2), rgb_image)

    # Draw all boxes onto image.
    for box, text in box_to_display_str_map.items():
        #print("{} :{}".format(box,text[0]))
        ymin, xmin, ymax, xmax = box
        im_height, im_width = rgb_image.shape[:2]
        if use_normalized_coordinates:
            (x1, x2, y1, y2) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))

            draw_bounding_box((x1, y1, x2, y2), rgb_image)
            draw_text((x1, y1, x2, y2), rgb_image, text[0])
    
    # print("the labels are: {}".format(box_to_display_str_map))

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
