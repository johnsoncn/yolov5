# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/10 15:30
@Author  : Johnson
@FileName: inference.py
"""

import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')
sys.path.append('../')
from norfair import Detection, Tracker
import torch

# Helper to convert bbox in format [x_min, y_min, x_max, y_max, score] to norfair.Detection class
def to_norfair(detects, frame_id):
    result = []
    for x_min, y_min, x_max, y_max, score in detects:
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        result.append(Detection(points=np.array([xc, yc]), scores=np.array([score]), data=np.array([w, h, frame_id])))

    return result


# Euclidean distance function to match detections on this frame with tracked_objects from previous frames
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def tracking_function(tracker, frame_id, bboxes, scores):
    detects = []
    predictions = []

    if len(scores) > 0:
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = scores[i]
            x_min = int(box[0])
            y_min = int(box[1])
            bbox_width = int(box[2])
            bbox_height = int(box[3])
            detects.append([x_min, y_min, x_min + bbox_width, y_min + bbox_height, score])
            predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))
    #             print(predictions[:-1])
    # Update tracks using detects from current frame
    tracked_objects = tracker.update(detections=to_norfair(detects, frame_id))
    for tobj in tracked_objects:
        bbox_width, bbox_height, last_detected_frame_id = tobj.last_detection.data
        if last_detected_frame_id == frame_id:  # Skip objects that were detected on current frame
            continue
        # Add objects that have no detections on current frame to predictions
        xc, yc = tobj.estimate[0]
        x_min, y_min = int(round(xc - bbox_width / 2)), int(round(yc - bbox_height / 2))
        score = tobj.last_detection.scores[0]

        predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))

    return predictions


tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=30,
    hit_inertia_min=3,
    hit_inertia_max=6,
    initialization_delay=1,
)


def voc2yolo(bboxes, image_height=720, image_width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]

    bboxes[..., 0] = bboxes[..., 0] + w / 2
    bboxes[..., 1] = bboxes[..., 1] + h / 2
    bboxes[..., 2] = w
    bboxes[..., 3] = h

    return bboxes


def yolo2voc(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


def coco2yolo(bboxes, image_height=720, image_width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # normolizinig
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2

    return bboxes


def yolo2coco(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    # denormalizing
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    # converstion (xmid, ymid) => (xmin, ymin)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2

    return bboxes


def voc2coco(bboxes, image_height=720, image_width=1280):
    bboxes = voc2yolo(bboxes, image_height, image_width)
    bboxes = yolo2coco(bboxes, image_height, image_width)
    return bboxes

class Starfish(object):

    # IMG_SIZE  = 9000
    # CONF      = 0.25
    # IOU       = 0.40
    # AUGMENT   = True

    def __init__(self, chpt_path, hub_load):
        self.IMG_SIZE = 6400
        # self.ROOT_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'
        self.CKPT_PATH = chpt_path
        self.HUB_LOAD = hub_load

        self.model = self.load_model(self.CKPT_PATH)

    def load_model(self, ckpt_path, conf=0.25, iou=0.50):
        model = torch.hub.load(self.HUB_LOAD, # /kaggle/input/yolov5-lib-ds
                               'custom',
                               path=ckpt_path,
                               source='local',
                               force_reload=True)  # local repo
        model.conf = conf
        model.iou = iou
        return model

    def predict(self, model, img, size, augment=False):
        height, width = img.shape[:2]
        results = model(img, size=size, augment=augment)  # custom inference size
        preds = results.pandas().xyxy[0]

        bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
        if len(bboxes):
            bboxes = voc2coco(bboxes, height, width).astype(int)
            confs = preds.confidence.values
            return bboxes, confs
        else:
            return [], []

    def kaggle_env(self):
        import greatbarrierreef
        env = greatbarrierreef.make_env()  # initialize the environment
        iter_test = env.iter_test()  # an iterator which loops over the test set and sample submission
        return env, iter_test

    def test_local(self, dir):
        import cv2,os
        frame_id = 0
        sources = os.listdir(dir)
        for idx, src in enumerate(tqdm(sources)):
            img = cv2.imread(os.path.join(dir, src))[:, :, ::-1]
            bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=False)
            new_bboxes, new_confs = [], []
            for i, conf in enumerate(confs):
                if conf > 0.28:
                    new_bboxes.append(bboxes[i])
                    new_confs.append(conf)

            predictions = tracking_function(tracker, frame_id, new_bboxes, new_confs)
            prediction_str = ' '.join(predictions)
            frame_id += 1
            print(src, prediction_str)


    def test_kaggle_tracking(self):
        env, iter_test = self.kaggle_env()
        frame_id = 0
        for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
            bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=True)
            new_bboxes, new_confs = [], []
            for i, conf in enumerate(confs):
                if conf > 0.28:
                    new_bboxes.append(bboxes[i])
                    new_confs.append(conf)

            predictions = tracking_function(tracker, frame_id, new_bboxes, new_confs)
            prediction_str = ' '.join(predictions)
            pred_df['annotations'] = prediction_str

            env.predict(pred_df)
            frame_id += 1

    def read(self):
        sub_df = pd.read_csv('submission.csv')
        sub_df.head()
        print(sub_df)



# CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/10000_resolution/weights/10000.pt'
CKPT_PATH = '/home/dingchaofan/yolov5/johnson/kaggle/starfish/weights/f2_sub2.pt/f2_sub2.pt'
HUB_LOAD = '/home/dingchaofan/yolov5'

# CKPT_PATH = '../input/1110000/best.pt'
# HUB_LOAD = '/kaggle/input/yolov5-lib-ds'




starfish = Starfish(chpt_path=CKPT_PATH, hub_load=HUB_LOAD)

if __name__ == "__main__":


    # starfish.test_kaggle_tracking()
    starfish.test_local(dir = '/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train')
