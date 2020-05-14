import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class variable:##
    def __init__(self):
        self.r_sh = 0
        self.r_el = 0
        self.r_wr = 0
        self.l_sh = 0
        self.l_el = 0
        self.l_wr = 0
        self.length = 10
        self.r_target = 0
        self.l_target = 0

def gui(img, pos):##
    result = np.zeros(img.shape, dtype=np.uint8)
    height = int(img.shape[0]/2)
    width = int(img.shape[1]/2)
    result[int(height/2):int(height/2)+height, int(width/2):int(width/2)+width] = cv2.resize(img,(width, height))

    #length
    if not pos[0,0] == 0
        if not pos[3,0] == 0:
            vari.length = np.abs(pos[0,0]-pos[3,0])

    #right
    if not pos[2,0] == 0:
        vari.r_target = np.min([1, np.max([0, (pos[2,0]-pos[0,0])/vari.length])])#0-1 r_wr-r_sh
    elif not pos[1,0] == 0:
        vari.r_target = np.min([1, np.max([0, (pos[1,0]-pos[0,0])/vari.length])])#r_el-r_sh

    #left
    if not pos[5,0] == 0:
        vari.l_target = np.min([1, np.max([0, (pos[3,0]-pos[5,0])/vari.length])])#l_sh-l_wr 
    elif not pos[4,0] == 0:
        vari.l_target = np.min([1, np.max([0, (pos[3,0]-pos[4,0])/vari.length])])#l_sh-l_el

    #gui
    y_begin = int(img.shape[0]/2) - 5
    x_begin = int(img.shape[1]/2 + img.shape[1]/2 * vari.r_target)
    cv2.rectangle(result, (x_begin, y_begin), (x_begin+10, y_begin+5), (255, 255, 255), thickness=-1)#right
    x_begin = int(img.shape[1]/2 * vari.l_target)
    cv2.rectangle(result, (x_begin, y_begin), (x_begin+10, y_begin+5), (255, 255, 255), thickness=-1)#right

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    cv2.namedWindow("game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('game', 640, 480)##
    vari = variable()##
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        image = cv2.flip(image, 1)##

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image, pos = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image = gui(image, pos)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('game', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
