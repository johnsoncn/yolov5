# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/10 13:59
@Author  : Johnson
@FileName: infer.py
"""

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
from tools import *

ROOT_DIR  = '/home/dingchaofan/data/barrier_reef_data/dataset'
# CKPT_PATH = '/kaggle/input/greatbarrierreef-yolov5-train-ds/yolov5/runs/train/exp/weights/best.pt'
CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'


IMG_SIZE  = 1280
CONF      = 0.15
IOU       = 0.50
AUGMENT   = False

def get_path(row):
    row['image_path'] = f'{ROOT_DIR}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

# # Train Data
# df = pd.read_csv(f'{ROOT_DIR}/train.csv')
# df = df.progress_apply(get_path, axis=1)
# df['annotations'] = df['annotations'].progress_apply(lambda x: ast.literal_eval(x))
# print(df.head(2))


def load_model(ckpt_path, conf=0.25, iou=0.50):
    model = torch.hub.load('/home/dingchaofan/yolov5',  # /kaggle/input/yolov5-lib-ds
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model


def predict(model, img, size=768, augment=False):
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


model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
img = cv2.imread('/home/dingchaofan/data/barrier_reef_data/dataset/images/valid/0-56.jpg')[...,::-1]
bboxes, confis = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
print(bboxes, confis)
plt_img = show_img(img, bboxes, bbox_format='coco')
plt_img.show()
plt_img.close()


# image_paths = df[df.num_bbox>1].sample(100).image_path.tolist()

# for idx, path in enumerate(image_paths):
#     img = cv2.imread(path)[...,::-1]
#     bboxes, confis = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
#     print(bboxes)
#     show_img(img, bboxes, bbox_format='coco')
#     if idx>5:
#         break
#
# import greatbarrierreef
# env = greatbarrierreef.make_env()# initialize the environment
# iter_test = env.iter_test()      # an iterator which loops over the test set and sample submission
#
# model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
# for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
#     bboxes, confs  = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
#     annot          = format_prediction(bboxes, confs)
#     pred_df['annotations'] = annot
#     env.predict(pred_df)
#     if idx<3:
#         display(show_img(img, bboxes, bbox_format='coco'))
#
# sub_df = pd.read_csv('submission.csv')
# sub_df.head()