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
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')
sys.path.append('../')
import torch
from PIL import Image
import ast
from tools import *
from torchsummary import summary
from pseudo_tools import run_wbf, TTAImage, rotBoxes90, detect1Image
from ensemble_boxes import letterbox


# 0.626的训练不变！只是改了推理尺寸10000
# 然后用自己3600训的重新提交了下

# 如果分不错，那就把伪标签数据推理出来，重新训

class Starfish(object):

    def __init__(self):
        self.IMG_SIZE = 3888
        # self.ROOT_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/10000_resolution/weights/10000.pt'
        self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/l6_3600_uflip_vm5_f12_up/f1/best.pt'
        self.model = self.load_model(self.CKPT_PATH)

    def load_model(self, ckpt_path, conf=0.25, iou=0.50):
        model = torch.hub.load('/home/dingchaofan/yolov5',   # ../input/yolov5-lib-ds
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

    def format_prediction(self, bboxes, confs):
        annot = ''
        if len(bboxes) > 0:
            for idx in range(len(bboxes)):
                xmin, ymin, w, h = bboxes[idx]
                conf = confs[idx]
                annot += f'{conf} {xmin} {ymin} {w} {h}'
                annot += ' '
            annot = annot.strip(' ')
        else:
            print('[INFO] no bbox detected')
            # return None
        return annot

    def kaggle_env(self):
        import greatbarrierreef
        env = greatbarrierreef.make_env()  # initialize the environment
        iter_test = env.iter_test()  # an iterator which loops over the test set and sample submission
        return env, iter_test

    def test_kaggle_tracking(self):
        env, iter_test = self.kaggle_env()

        frame_id = 0
        for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
            bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=False)

            predictions = tracking_function(tracker, frame_id, bboxes, confs)
            prediction_str = ' '.join(predictions)
            pred_df['annotations'] = prediction_str

            env.predict(pred_df)
            frame_id += 1

    def test_kaggle(self):
        env, iter_test = self.kaggle_env()
        # for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
        #     bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=False)
        #     annot = self.format_prediction(bboxes, confs)
        #     print(type(annot))
        #     pred_df['annotations'] = annot
        #     env.predict(pred_df)

        for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
            anno = ''
            r = self.model(img, size=10000, augment=True)  # 训练还是3600，只是改了推理
            if r.pandas().xyxy[0].shape[0] == 0:
                anno = ''
            else:
                for idx, row in r.pandas().xyxy[0].iterrows():
                    if row.confidence > 0.28:
                        anno += '{} {} {} {} {} '.format(row.confidence, int(row.xmin), int(row.ymin),
                                                         int(row.xmax - row.xmin), int(row.ymax - row.ymin))
            #                 pred.append([row.confidence, row.xmin, row.ymin, row.xmax-row.xmin, row.ymax-row.ymin])
            pred_df['annotations'] = anno.strip(' ')
            env.predict(pred_df)

    def test_single(self, src, save):
        img = cv2.imread(src)[:, :, ::-1]
        # cv2.imshow('img1', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=False)
        annot = self.format_prediction(bboxes, confs)
        if annot != None and save == True:
            plt_img = show_img(img, bboxes, bbox_format='coco')
            plt_img.save('2.png')
            # plt_img.show()
            # plt_img.close()
        return annot

    def test_pseudo(self):
        is_TTA = True

        # unlabeled_dir = '/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/'
        unlabeled_dir = '/home/dingchaofan/dataset/starfish/data/labeled_data/images/train/'
        import os
        data_list = os.listdir(unlabeled_dir)

        for idx, path in enumerate(data_list):
            # img = cv2.imread(unlabeled_dir+path)[..., ::-1]
            # print(img.shape)
            # print(img.ndimension())
            # img = img.unsqueeze(0)
            # bboxes, confis = self.predict(self.model, img, size=self.IMG_SIZE, augment=True)


            img = cv2.imread(unlabeled_dir+path)
            boxes, scores = detect1Image(img, self.IMG_SIZE, self.model, 'cuda', 0.25, 0.5)

            print(boxes)


    def read(self):
        sub_df = pd.read_csv('submission.csv')
        sub_df.head()
        print(sub_df)


starfish = Starfish()

if __name__ == "__main__":

    from pseudo_labeling import TTAImage

    # src = '/home/dingchaofan/data/barrier_reef_data/dataset/images/valid/0-4237.jpg'
    # src = '/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/2-191.jpg'

    # import os
    # data_list = os.listdir('/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/')
    # for src in data_list:
    #
    #     annot = starfish.test_single('/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/'+src, show=True)
    #     print(src, annot)

    # annot = starfish.test_single(src, save=False)
    # print(annot)

    starfish.test_pseudo()

    # starfish.test_kaggle()
    # starfish.read()