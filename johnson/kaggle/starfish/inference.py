# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/10 15:30
@Author  : Johnson
@FileName: inference.py
"""
import time
from tqdm import tqdm
# tqdm.pandas()
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
import numpy as np
# from torchsummary import summary
# from pseudo_tools import run_wbf, TTAImage, rotBoxes90, detect1Image
# from ensemble_boxes import letterbox


# 0.626的训练不变！只是改了推理尺寸10000
# 然后用自己3600训的重新提交了下

# 如果分不错，那就把伪标签数据推理出来，重新训

class Starfish(object):

    def __init__(self):
        self.IMG_SIZE = 10500
        # self.ROOT_DIR = '/home/dingchaofan/data/barrier_reef_data/dataset'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/10000_resolution/weights/10000.pt'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/runs/train/l6_3600_uflip_vm5_f12_up/f1/best.pt'
        # self.CKPT_PATH = '/home/dingchaofan/yolov5/johnson/kaggle/starfish/weights/f2_sub2.pt/f2_sub2.pt'
        self.CKPT_PATH = 'f2_sub2.pt'
        self.model = self.load_model(self.CKPT_PATH)

    def load_model(self, ckpt_path, conf=0.60, iou=0.50):
        model = torch.hub.load(f'C:\\Users\\dingchaofan\\Desktop\\yolov5',   # ../input/yolov5-lib-ds # C:\Users\dingchaofan\Desktop\yolov5 # /home/dingchaofan/yolov5
                               'custom',
                               path=ckpt_path,
                               source='local',
                               force_reload=True)  # local repo
        model.conf = conf
        model.iou = iou
        return model

    def predict(self, model, img, size, augment=False, pseudo=False):

        height, width = img.shape[:2]
        results = model(img, size=size, augment=augment)  # custom inference size
        preds = results.pandas().xyxy[0]
        bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values
        if len(bboxes):
            if pseudo:
                # transfer to yolo format if predicting pseudo label
                bboxes = voc2yolo(bboxes, height, width)
            else:
                # normal prediction (coco format)
                bboxes = voc2coco(bboxes, height, width)
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
            return False
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
        torch.cuda.synchronize()
        start = time.time()

        bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=False)
        torch.cuda.synchronize()
        end = time.time()
        print(end-start)

        annot = self.format_prediction(bboxes, confs)
        if annot and save == True:
            plt_img = show_img(img, bboxes, bbox_format='coco')
            plt_img.save('2.png')
            # plt_img.show()
            # plt_img.close()
        return annot

    def test_pseudo(self, save_txt=True):
        import os

        root_dir = '/home/dingchaofan/dataset/starfish/data/pseudo_data/'

        unlabeled_images = root_dir + 'images/unlabeled_images/train/'
        label_save_to = root_dir + 'labels/train/'

        # results_with_label = root_dir + 'test_results3'
        os.makedirs(label_save_to, exist_ok=True)

        import os
        data_list = os.listdir(unlabeled_images)

        for idx in tqdm(range(len(data_list))):
            img = cv2.imread(unlabeled_images + data_list[idx])
            bboxes, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=True, pseudo=True)
            annot = self.format_prediction(bboxes, confs)

            if annot and save_txt == True:
                print(data_list[idx])

                name = data_list[idx].replace('jpg', 'txt')
                print(name, annot)

                with open(label_save_to + name, "w", encoding="utf-8") as f:
                    f.write(annot)

                # plt_img = show_img(img, bboxes, bbox_format='yolo')
                # plt_img.save(f'{results_with_label}/{data_list[idx]}.png')

    def evaluate_f2(self):


        image_files = '/home/dingchaofan/dataset/starfish/data/labeled_data/images/train/'
        label_txt = '/home/dingchaofan/dataset/starfish/data/labeled_data/labels/train/'

        data_list = os.listdir(image_files)


        from f2_metric import f2, IOU_coco
        # Which IOU level to evaluate (Competition metric tests 0.3 to 0.8 with step of 0.05)
        eval_IOU = 0.30

        # Confidence scores of true positives, false positives and count false negatives
        TP = []  # Confidence scores of true positives
        FP = []  # Confidence scores of true positives
        FN = 0  # Count of false negative boxes

        for idx in tqdm(range(len(data_list[:100]))):
            # print(image_files + data_list[idx])
            img = cv2.imread(image_files + data_list[idx])
            preds, confs = self.predict(self.model, img, size=self.IMG_SIZE, augment=True)

            if len(preds):
                preds = coco2yolo(np.array(preds))
            # else:
            #     preds = np.array([])
            # print(bboxes)

            with open(label_txt + data_list[idx].replace('jpg', 'txt'), "r", encoding="utf-8") as f:
                bboxes = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                if len(bboxes):  # 误"w"到之前的文件。先判断下
                    bboxes = np.delete(bboxes, 0, axis=1)
                    # print(preds)

            # Test YOLOV5
            gt0 = bboxes.tolist()
            if len(preds) == 0:
                # all gt are false negative
                FN += len(gt0)
            else:
                bb = preds.copy().tolist()
                for idx, b in enumerate(bb):
                    b.append(confs[idx])
                bb.sort(key=lambda x: x[4], reverse=True)

                if len(gt0) == 0:
                    # all bboxes are false positives
                    for b in bb:
                        FP.append(b[4])
                else:
                    # match bbox with gt
                    for b in bb:
                        matched = False
                        for g in gt0:
                            # check whether gt box is already matched to an inference bb
                            if len(g) == 4:
                                # g bbox is unmatched
                                if IOU_coco(b, g) >= eval_IOU:
                                    g.append(b[4])  # assign confidence values to g; marks g as matched
                                    matched = True
                                    TP.append(b[4])
                                    break
                        if not matched:
                            FP.append(b[4])
                    for g in gt0:
                        if len(g) == 4:
                            FN += 1

        print(f'True positives = {len(TP)}')
        print(f'False negatives = {FN}')

        F2list = []
        F2max = 0.0
        F2maxat = -1.0

        for c in np.arange(0.0, 1.0, 0.01):
            FNcount = FN + sum(1 for i in TP if i < c)
            TPcount = sum(1 for i in TP if i >= c)
            FPcount = sum(1 for i in FP if i >= c)
            R = TPcount / (TPcount + FNcount + 0.0001)
            P = TPcount / (TPcount + FPcount + 0.0001)
            F2 = (5 * P * R) / (4 * P + R + 0.0001)
            F2list.append((c, F2))
            if F2max < F2:
                F2max = F2
                F2maxat = c

        print(f'F2 max is {F2max} at CONF = {F2maxat}')


    def read(self):
        sub_df = pd.read_csv('submission.csv')
        sub_df.head()
        print(sub_df)


starfish = Starfish()

if __name__ == "__main__":

    # from pseudo_labeling import TTAImage

    # src = '/home/dingchaofan/data/barrier_reef_data/dataset/images/valid/0-4237.jpg'
    # src = '/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/2-191.jpg'

    # import os
    # data_list = os.listdir('/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/')
    # for src in data_list:
    #
    #     annot = starfish.test_single('/home/dingchaofan/dataset/starfish/data/pseudo_data/images/unlabeled_images/train/'+src, show=True)
    #     print(src, annot)

    # annot = starfish.test_single(src, save=False)

    import sys
    print(sys.platform)

    # print(annot)

    # starfish.evaluate_f2()
    while True:
        annot = starfish.test_single(src= '0e908864493871.5b028d5c66a8a.png', save=False)
        print(annot)

    # starfish.test_kaggle()
    # starfish.read()