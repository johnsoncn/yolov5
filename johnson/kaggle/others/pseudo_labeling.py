# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/21 10:34
@Author  : Johnson
@FileName: pseudo_labeling.py
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh
#
# from utils.datasets import *
# from utils.utils import *
from pseudo_tools import run_wbf, TTAImage, rotBoxes90, detect1Image
import torch

def makePseudolabel():
    source = '/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/'
    weights = '/home/dingchaofan/yolov5/runs/train/l6_3600_uflip_vm5_f12_up/f1/best.pt'
    imgsz = 1024
    conf_thres = 0.5
    iou_thres = 0.6
    is_TTA = True

    imagenames = os.listdir(source)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    # dataset = LoadImages(source, img_size=imgsz)

    path2save = '/home/dingchaofan/dataset/starfish/data/pseudo_data/'
    if not os.path.exists(path2save + 'images/train'):
        os.makedirs(path2save + 'images/train')
    if not os.path.exists(path2save + 'labels/train'):
        os.makedirs(path2save + 'labels/train')

    for name in imagenames:
        image_id = name.split('.')[0]
        print(image_id)

    # for name in imagenames:
    #     image_id = name.split('.')[0]
    #     im01 = cv2.imread('%s/%s.jpg' % (source, image_id))  # BGR
    #     if im01.shape[0] != 1024 or im01.shape[1] != 1024:
    #         continue
    #     assert im01 is not None, 'Image Not Found '
    #     # Padded resize
    #     im_w, im_h = im01.shape[:2]
    #     if is_TTA:
    #         enboxes = []
    #         enscores = []
    #         for i in range(4):
    #             im0 = TTAImage(im01, i)
    #             boxes, scores = detect1Image(im0, imgsz, model, device, conf_thres, iou_thres)
    #             for _ in range(3 - i):
    #                 boxes = rotBoxes90(boxes, im_w, im_h)
    #
    #             enboxes.append(boxes)
    #             enscores.append(scores)
    #
    #         boxes, scores, labels = run_wbf(enboxes, enscores, image_size=im_w, iou_thr=0.6, skip_box_thr=0.43)
    #         boxes = boxes.astype(np.int32).clip(min=0, max=im_w)
    #     else:
    #         boxes, scores = detect1Image(im01, imgsz, model, device, conf_thres, iou_thres)
    #
    #     boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    #     boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    #
    #     boxes = boxes[scores >= 0.05].astype(np.int32)
    #     scores = scores[scores >= float(0.05)]
    #
    #     lineo = ''
    #     for box in boxes:
    #         x1, y1, w, h = box
    #         xc, yc, w, h = (x1 + w / 2) / 1024, (y1 + h / 2) / 1024, w / 1024, h / 1024
    #         lineo += '0 %f %f %f %f\n' % (xc, yc, w, h)
    #
    #     # fileo = open('convertor/fold0/labels/' + path2save + image_id + ".txt", 'w+')
    #     # fileo.write(lineo)
    #     # fileo.close()
    #     # sh.copy("../input/global-wheat-detection/test/{}.jpg".format(image_id),
    #     #         'convertor/fold0/images/{}/{}.jpg'.format(path2save, image_id))
    #
    #     fileo = open(path2save + 'labels/train/'+ image_id + ".txt", 'w+')
    #     fileo.write(lineo)
    #     fileo.close()
    #     sh.copy("../input/global-wheat-detection/test/{}.jpg".format(image_id),
    #             'convertor/fold0/images/{}/{}.jpg'.format(path2save, image_id))

makePseudolabel()

