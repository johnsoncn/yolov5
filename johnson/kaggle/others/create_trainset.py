# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/6 18:33
@Author  : Johnson
@FileName: starfish_dataloader.py
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import groupby
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import os
import pickle
import cv2
from multiprocessing import Pool
import matplotlib.pyplot as plt
# import cupy as cp
import ast
import glob
from os import listdir
from os.path import isfile, join
from glob import glob
import yaml

import shutil
from shutil import copyfile
import sys

from joblib import Parallel, delayed

from tools import *
# --- Read data ---
TRAIN_PATH = '/home/dingchaofan/data/barrier_reef_data/dataset'

def get_path(row):
    row['image_path'] = f'{TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row

BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE=1280
Selected_Fold=4  #0 1 2 3 4

# Read in the data CSV files
df = pd.read_csv(f"{TRAIN_PATH}/train.csv")
print('total bbox :', len(df)) # 23501

# 统计每一行的bbox数量，并新增一列以NumBBox表示
df["NumBBox"]=df['annotations'].apply(lambda x: str.count(x, 'x'))
# print(df.head(5))

# 筛选有标注的作为训练集
df_train=df[df["NumBBox"]>0]
print('with bbox :', len(df_train)) # 4919
print('有bbox只占总体的 :', len(df_train) / len(df))
# print(df_train)
# df_train.sample(2)

df_train['annotations'] = df_train['annotations'].progress_apply(lambda x: ast.literal_eval(x))
df_train['bboxes'] = df_train.annotations.progress_apply(get_bbox)
print(df_train)

# print(df.loc[
#           df['video_frame']==9077
#           ])

# All images have Width=1280 & Height=720
df_train["Width"]=1280
df_train["Height"]=720
print(df_train.sample(2))

df_train = df_train.progress_apply(get_path, axis=1)


def plot_bbox():
    from tools import load_image, coco2yolo, draw_bboxes, colors

    df_v = df_train[(df_train.NumBBox == 13)].sample(2)
    fig, ax = plt.subplots(1, 2, figsize=(30, 20))
    i = 0
    for index, row in df_v.iterrows():
        img = load_image(row.image_path)
        image_height = row.Height
        image_width = row.Width
        bboxes_coco = np.array(row.bboxes)
        bboxes_yolo = coco2yolo(image_height, image_width, bboxes_coco)
        names = ['starfish'] * len(bboxes_coco)
        labels = [0] * len(bboxes_coco)
        im = draw_bboxes(img=img,
                         bboxes=bboxes_yolo,
                         classes=names,
                         class_ids=labels,
                         class_name=True,
                         colors=colors,
                         bbox_format='yolo',
                         line_thickness=2)
        ax[i].imshow(im)
        ax[i].axis('OFF')
        i = i + 1
    # plt.show()
    # plt.close()

from sklearn.model_selection import GroupKFold
# """
# 0    1100
# 3     970
# 2     968
# 4     945
# 1     936
# """
kf = GroupKFold(n_splits = 5)
df_train = df_train.reset_index(drop=True)
df_train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y = df_train.video_id.tolist(), groups=df_train.sequence)):
    df_train.loc[val_idx, 'fold'] = fold
print(df_train.sample(2))
# df_train.to_csv('my_csv.csv',encoding='utf-8',index=False)
#
#
# print(df_train.fold.value_counts())

SAVE_TO =  '/home/dingchaofan/data/barrier_reef_data/fulldata'

# create train & val images
os.makedirs(f'{SAVE_TO}/images/train', exist_ok=True)
# os.makedirs(f'{SAVE_TO}/images/valid', exist_ok=True)
os.makedirs(f'{SAVE_TO}/labels/train', exist_ok=True)
# os.makedirs(f'{SAVE_TO}/labels/valid', exist_ok=True)
#
# for i in tqdm(range(len(df_train))):
#     row = df_train.loc[i]
#     if row.fold != Selected_Fold:
#         copyfile(f'{row.image_path}', f'{TRAIN_PATH}/images/train/{row.image_id}.jpg')
#     else:
#         copyfile(f'{row.image_path}', f'{TRAIN_PATH}/images/valid/{row.image_id}.jpg')


# for i in tqdm(range(len(df_train))):
#     row = df_train.loc[i]
#     # if row.fold != Selected_Fold:
#     copyfile(f'{row.image_path}', f'{SAVE_TO}/images/train/{row.image_id}.jpg')


    # else:
    #     copyfile(f'{row.image_path}', f'{TRAIN_PATH}/images/valid/{row.image_id}.jpg')

"""
yolo: 中心点 + 宽高      (x_center, y_center, w, h)
coco: 左上坐标点 + 宽高  (x_left_top, y_left_top, w, h
"""
def create_labels():
    from tools import coco2yolo

    all_bboxes = []
    for row_idx in tqdm(range(df_train.shape[0])):
        row = df_train.iloc[row_idx]
        # Get image
        image_name = row.image_id
        image_height = row.Height
        image_width  = row.Width
        bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
        num_bbox     = len(bboxes_coco)
        names        = ['cots']*num_bbox
        labels       = [0]*num_bbox
        # if row.fold != Selected_Fold:
        #     file_name = f'{TRAIN_PATH}/labels/train/{image_name}.txt'
        # else:
        #     file_name = f'{TRAIN_PATH}/labels/valid/{image_name}.txt'
        file_name = f'{SAVE_TO}/labels/train/{image_name}.txt'

        with open(file_name, 'w') as f:
            print(image_height, image_width, bboxes_coco)
            bboxes_yolo  = coco2yolo(bboxes_coco, image_height, image_width)
            bboxes_yolo  = np.clip(bboxes_yolo, 0, 1)
            all_bboxes.extend(bboxes_yolo)
            for bbox_idx in range(len(bboxes_yolo)):
                bb=str(bboxes_yolo[bbox_idx])
                bb=bb[1:-1]
                #annot = [str(labels[bbox_idx])]+ list(bboxes_yolo[bbox_idx].astype(str))+(['\n'] if num_bbox!=(bbox_idx+1) else [''])
                annot = str(str(labels[bbox_idx])) + ' ' + bb + '\n'
                annot = ''.join(annot)
                annot = annot.strip('')
                f.write(annot)

create_labels()
# 修改下map 0.3 0.8 0.95？