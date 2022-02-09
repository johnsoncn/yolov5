# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/6 16:25
@Author  : Johnson
@FileName: barrier_reef_dataloader.py
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

from joblib import Parallel, delayed

# from IPython.display import display, HTML

from matplotlib import animation, rc
rc('animation', html='jshtml')


"""
train_images/ - Folder containing training set photos of the form video_{video_id}/{video_frame}.jpg.

[train/test].csv - Metadata for the images. As with other test files, most of the test metadata data is only available to your notebook upon submission. Just the first few rows available for download.

video_id - ID number of the video the image was part of. The video ids are not meaningfully ordered.

video_frame - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.
sequence - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.
sequence_frame - The frame number within a given sequence.
image_id - ID code for the image, in the format {video_id}-{video_frame}
annotations - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. 
            Does not use the same format as the predictions you will submit. Not available in test.csv. 
            A bounding box is described by the pixel coordinate (x_min, y_min) of its lower left corner within the image together with its width and height in pixels --> (COCO format).

"""

FOLD      = 4 # which fold to train
DIM       = 1280
MODEL     = 'yolov5m'
BATCH     = 8
EPOCHS    = 25

PROJECT   = 'great-barrier-reef-public' # w&b in yolov5
NAME      = f'{MODEL}-dim{DIM}-fold{FOLD}' # w&b for yolov5
ROOT_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset'

REMOVE_NOBBOX = True # remove images with no bbox
# ROOT_DIR  = '/kaggle/input/tensorflow-great-barrier-reef/'
IMAGE_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset/images' # directory to save images
LABEL_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset/labels' # directory to save labels

# Train Data
df = pd.read_csv(f'{ROOT_DIR}/train.csv')
df['old_image_path'] = f'{ROOT_DIR}/train_images/video_'+df.video_id.astype(str)+'/'+df.video_frame.astype(str)+'.jpg'
df['image_path']  = f'{IMAGE_DIR}/'+df.image_id+'.jpg'
df['label_path']  = f'{LABEL_DIR}/'+df.image_id+'.txt'
df['annotations'] = df['annotations'].progress_apply(eval)
print(df.head(2))
# display(df.head(2))

# Nearly 80% images are without any bbox.
df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.num_bbox>0).value_counts(normalize=True)*100
print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")

if REMOVE_NOBBOX:
    df = df.query("num_bbox>0")

def make_copy(row):
    shutil.copyfile(row.old_image_path, row.image_path)
    return

# image_paths = df.old_image_path.tolist()
# _ = Parallel(n_jobs=-1, backend='threading')(delayed(make_copy)(row) for _, row in tqdm(df.iterrows(), total=len(df)))


# check https://github.com/awsaf49/bbox for source code of following utility functions
# from bbox.utils import coco2yolo, coco2voc, voc2yolo
# from bbox.utils import draw_bboxes, load_image
# from bbox.utils import clip_bbox, str2annot, annot2str
from bbox_utils import coco2voc, clip_bbox, voc2yolo, annot2str

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))\
          for idx in range(1)]

df['bboxes'] = df.annotations.progress_apply(get_bbox)
df.head(2)

df['width']  = 1280
df['height'] = 720
print(df.head(2))

cnt = 0
all_bboxes = []
bboxes_info = []
for row_idx in tqdm(range(df.shape[0])):
    row = df.iloc[row_idx]
    image_height = row.height
    image_width  = row.width
    bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
    num_bbox     = len(bboxes_coco)
    names        = ['cots']*num_bbox
    labels       = np.array([0]*num_bbox)[..., None].astype(str)
    ## Create Annotation(YOLO)
    with open(row.label_path, 'w') as f:
        if num_bbox<1:
            annot = ''
            f.write(annot)
            cnt+=1
            continue
        bboxes_voc  = coco2voc(bboxes_coco, image_height, image_width)
        bboxes_voc  = clip_bbox(bboxes_voc, image_height, image_width)
        bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
        all_bboxes.extend(bboxes_yolo.astype(float))
        bboxes_info.extend([[row.image_id, row.video_id, row.sequence]]*len(bboxes_yolo))
        annots = np.concatenate([labels, bboxes_yolo], axis=1)
        string = annot2str(annots)
        f.write(string)
print('Missing:',cnt)