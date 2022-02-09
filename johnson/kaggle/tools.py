
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
    # print('bboxes', bboxes)
    # bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

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


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bboxes(img, bboxes, classes, class_ids, colors=None, show_classes=None, bbox_format='yolo', class_name=False,
                line_thickness=2):
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255, 0) if colors is None else colors

    if bbox_format == 'yolo':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = round(float(bbox[0]) * image.shape[1])
                y1 = round(float(bbox[1]) * image.shape[0])
                w = round(float(bbox[2]) * image.shape[1] / 2)  # w/2
                h = round(float(bbox[3]) * image.shape[0] / 2)

                voc_bbox = (x1 - w, y1 - h, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(get_label(cls)),
                             line_thickness=line_thickness)

    elif bbox_format == 'coco':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w = int(round(bbox[2]))
                h = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1 + w, y1 + h)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)

    elif bbox_format == 'voc_pascal':

        for idx in range(len(bboxes)):

            bbox = bboxes[idx]
            cls = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors

            if cls in show_classes:
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox,
                             image,
                             color=color,
                             label=cls if class_name else str(cls_id),
                             line_thickness=line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image



def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row


np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) \
          for idx in range(1)]