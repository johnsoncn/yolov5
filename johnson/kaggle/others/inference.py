# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/7 14:16
@Author  : Johnson
@FileName: inference.py
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')
import torch
from PIL import Image
import ast
from copy import copy
from tools import *


IMG_SIZE=1280
TRAIN_PATH = '/kaggle/input/tensorflow-great-barrier-reef'
# Best_Model = '/kaggle/input/barrie-reef-yolo5/yolov5/kaggle-Reef/exp/weights/best.pt'
Best_Model = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'
# runs/train/exp34/weights/best.pt


def load_model(Best_Model, conf=0.25, iou=0.50):
    model = torch.hub.load('/home/dingchaofan/yolov5',  # /kaggle/input/barrie-reef-yolo5/yolov5
                           'custom',
                           path=Best_Model,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    return model


def predict(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference size
    preds = results.pandas().xyxy[0]
    bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
    print(bboxes)
    if len(bboxes):
        bboxes = voc2coco(bboxes, height, width).astype(int)
        confs = preds.confidence.values
        return bboxes, confs
    else:
        return [], []


def format_prediction(bboxes, confs):
    annot = ''
    if len(bboxes) > 0:
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf = confs[idx]
            annot += f'{conf} {xmin} {ymin} {w} {h}'
            annot += ' '
        annot = annot.strip(' ')
    return annot


def show_img(img, bboxes, bbox_format='yolo'):
    names = ['starfish'] * len(bboxes)
    labels = [0] * len(bboxes)
    img = draw_bboxes(img=img,
                      bboxes=bboxes,
                      classes=names,
                      class_ids=labels,
                      class_name=True,
                      colors=colors,
                      bbox_format=bbox_format,
                      line_thickness=2)
    return Image.fromarray(img).resize((800, 400))

import greatbarrierreef
env = greatbarrierreef.make_env()# initialize the environment
iter_test = env.iter_test()      # an iterator which loops over the test set and sample submission



print(iter_test)
CONF= 0.15
IOU= 0.50
model = load_model(Best_Model, conf=CONF, iou=IOU)
for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
    # print(idx, (img, pred_df))
    bboxes, confs  = predict(model, img, size=IMG_SIZE, augment=True)
    print(bboxes, confs)
    annot          = format_prediction(bboxes, confs)
    print(annot)
    pred_df['annotations'] = annot
    env.predict(pred_df)
    if idx<3:
        print(show_img(img, bboxes, bbox_format='coco'))

sub_df = pd.read_csv('submission.csv')
sub_df.head()