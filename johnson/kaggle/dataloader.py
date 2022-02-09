# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/6 18:33
@Author  : Johnson
@FileName: starfish_dataloader.py
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import os
import ast
from shutil import copyfile
from sklearn.model_selection import GroupKFold
from tools import coco2yolo


class DataCreator(object):

    def __init__(self, create_unlabeled):
        # self.TRAIN_PATH = '/home/dingchaofan/data/barrier_reef_data/dataset'
        self.TRAIN_PATH = '/home/dingchaofan/dataset/starfish/data'
        self.BATCH_SIZE = 16
        self.EPOCHS = 30
        self.IMG_SIZE = 1280
        self.Selected_Fold = 4  # 0 1 2 3 4
        self.create_unlabeled = create_unlabeled

        if not self.create_unlabeled:

            self.df = pd.read_csv(f"{self.TRAIN_PATH}/train.csv")
            self.df_train = self.create_traindf() # add NumBBox bboxes image_path
            self.df_train = self.create_kflod(self.df_train)  # add fold

            # self.SAVE_TO = '/home/dingchaofan/data/barrier_reef_data/fulldata'
            self.SAVE_TO = '/home/dingchaofan/dataset/starfish/data/labeled_data'

            # print(self.df_train)
            # self.df_train.to_csv('labeled_data.csv', encoding='utf-8', index=False)

            self.create_path()
            self.create_images(self.df_train)
            self.create_labels(self.df_train)

        else:
            self.df = pd.read_csv(f"{self.TRAIN_PATH}/train.csv")
            self.df_train = self.create_unlabeldf()

            self.SAVE_TO = '/home/dingchaofan/dataset/starfish/data/pseudo_data'

            # self.create_path()
            self.create_unlabeled_images(self.df_train)
            print('done !!!')



    def create_unlabeldf(self):
        # 统计每一行的bbox数量，并新增一列以NumBBox表示
        self.df["NumBBox"] = self.df['annotations'].apply(lambda x: str.count(x, 'x'))

        # 筛选无标注的数据
        self.df_train = self.df[self.df["NumBBox"] == 0]
        print(f'[INFO] without bbox / total bbox = {len(self.df_train)} / {len(self.df)}')

        # print(self.df_train)

        # 添加一列image_path
        def get_path(row):
            row['image_path'] = f'{self.TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
            return row

        self.df_train = self.df_train.progress_apply(get_path, axis=1)
        # self.df_train.pop('annotations')
        # self.df_train.pop('NumBBox')
        return self.df_train

        # print(self.df_train)


    def create_traindf(self):
        # 统计每一行的bbox数量，并新增一列以NumBBox表示
        self.df["NumBBox"] = self.df['annotations'].apply(lambda x: str.count(x, 'x'))

        # 筛选有标注的作为训练集
        self.df_train = self.df[self.df["NumBBox"] > 0]
        print(f'[INFO] with bbox / total bbox = {len(self.df_train)} / {len(self.df)}')

        # 添加一列bboxes ： [{'x': 559, 'y': 213, 'width': 50, 'height': 32}] --> [[559, 213, 50, 32]]
        def get_bbox(annots):
            bboxes = [list(annot.values()) for annot in annots]
            return bboxes
        self.df_train['annotations'] = self.df_train['annotations'].progress_apply(lambda x: ast.literal_eval(x))
        self.df_train['bboxes'] = self.df_train.annotations.progress_apply(get_bbox)

        # 添加一列image_path
        def get_path(row):
            row['image_path'] = f'{self.TRAIN_PATH}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
            return row
        self.df_train = self.df_train.progress_apply(get_path, axis=1)

        return self.df_train

    def create_kflod(self, df_train):
        """
        # 0    1100
        # 3     970
        # 2     968
        # 4     945
        # 1     936
        :return: 划分flod，给每一行添加一个分配的 flod 编号 （0 1 2 3 4）
        """

        kf = GroupKFold(n_splits=5)
        self.df_train = df_train.reset_index(drop=True)
        self.df_train['fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(
                kf.split(self.df_train, y=self.df_train.video_id.tolist(), groups=self.df_train.sequence)):
            # 添加一列 flod 编号
            self.df_train.loc[val_idx, 'fold'] = fold

        # print(self.df_train.fold.value_counts())
        # df_train.to_csv('my_csv.csv',encoding='utf-8',index=False)
        return self.df_train

    def create_path(self):
        if not self.create_unlabeled:
            # create train & val images
            os.makedirs(f'{self.SAVE_TO}/images/train', exist_ok=True)
            os.makedirs(f'{self.SAVE_TO}/images/valid', exist_ok=True)
            os.makedirs(f'{self.SAVE_TO}/labels/train', exist_ok=True)
            os.makedirs(f'{self.SAVE_TO}/labels/valid', exist_ok=True)
        else:
            os.makedirs(f'{self.SAVE_TO}/images/train', exist_ok=True)
        print('[INFO] CREATE PATH done !')

    def create_images(self, df_train):
        # df_train = df_train.reset_index(drop=True)
        print('[INFO] STARTING CREATE images ...')
        for i in tqdm(range(len(df_train))):
            row = df_train.loc[i]

            # if row.fold != self.Selected_Fold:
            #     copyfile(f'{row.image_path}', f'{self.SAVE_TO}/images/train/{row.image_id}.jpg')
            # else:
            #     copyfile(f'{row.image_path}', f'{self.SAVE_TO}/images/valid/{row.image_id}.jpg')

            copyfile(f'{row.image_path}', f'{self.SAVE_TO}/images/train/{row.image_id}.jpg')
            if row.fold == self.Selected_Fold:
                copyfile(f'{row.image_path}', f'{self.SAVE_TO}/images/valid/{row.image_id}.jpg')

    def create_unlabeled_images(self, df_train):
        print('[INFO] STARTING CREATE images ...')
        for ind, row in tqdm(df_train.iterrows()):
            # print(row['image_path'])
            copyfile(f'{row.image_path}', f'{self.SAVE_TO}/images/train/{row.image_id}.jpg')


    def create_labels(self, df_train):
        print('[INFO] STARTING CREATE labels ...')
        df_train["Width"] = 1280
        df_train["Height"] = 720

        all_bboxes = []
        for row_idx in tqdm(range(df_train.shape[0])):
            row = df_train.iloc[row_idx]
            # Get image
            image_name = row.image_id
            image_height = row.Height
            image_width = row.Width
            bboxes_coco = np.array(row.bboxes).astype(np.float32).copy()
            num_bbox = len(bboxes_coco)
            names = ['cots'] * num_bbox
            labels = [0] * num_bbox

            # if row.fold != Selected_Fold:
            #     file_name = f'{TRAIN_PATH}/labels/train/{image_name}.txt'
            # else:
            #     file_name = f'{TRAIN_PATH}/labels/valid/{image_name}.txt'

            file_name = f'{self.SAVE_TO}/labels/train/{image_name}.txt'
            # if row.fold == self.Selected_Fold:
            #     file_name = f'{self.SAVE_TO}/labels/valid/{image_name}.txt'

            with open(file_name, 'w') as f:
                bboxes_yolo = coco2yolo(bboxes_coco, image_height, image_width)
                bboxes_yolo = np.clip(bboxes_yolo, 0, 1)
                all_bboxes.extend(bboxes_yolo)
                for bbox_idx in range(len(bboxes_yolo)):
                    bb = str(bboxes_yolo[bbox_idx])
                    bb = bb[1:-1]
                    # annot = [str(labels[bbox_idx])]+ list(bboxes_yolo[bbox_idx].astype(str))+(['\n'] if num_bbox!=(bbox_idx+1) else [''])
                    annot = str(str(labels[bbox_idx])) + ' ' + bb + '\n'
                    annot = ''.join(annot)
                    annot = annot.strip('')
                    f.write(annot)

            if row.fold == self.Selected_Fold:
                val_file = f'{self.SAVE_TO}/labels/valid/{image_name}.txt'
                with open(val_file, 'w') as f:
                    bboxes_yolo = coco2yolo(bboxes_coco, image_height, image_width)
                    bboxes_yolo = np.clip(bboxes_yolo, 0, 1)
                    all_bboxes.extend(bboxes_yolo)
                    for bbox_idx in range(len(bboxes_yolo)):
                        bb = str(bboxes_yolo[bbox_idx])
                        bb = bb[1:-1]
                        # annot = [str(labels[bbox_idx])]+ list(bboxes_yolo[bbox_idx].astype(str))+(['\n'] if num_bbox!=(bbox_idx+1) else [''])
                        annot = str(str(labels[bbox_idx])) + ' ' + bb + '\n'
                        annot = ''.join(annot)
                        annot = annot.strip('')
                        f.write(annot)




if __name__ == '__main__':

    data_creator = DataCreator(create_unlabeled=False)

