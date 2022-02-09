


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


        self.df = pd.read_csv(f"{self.TRAIN_PATH}/train.csv")
        self.df_train = self.create_traindf() # add NumBBox bboxes image_path
        # self.df_train = self.create_kflod(self.df_train)  # add fold

        # self.SAVE_TO = '/home/dingchaofan/data/barrier_reef_data/fulldata'
        self.SAVE_TO = '/home/dingchaofan/dataset/starfish/data/labeled_data'

        print(self.df_train)
        # self.df_train.to_csv('labeled_data.csv', encoding='utf-8', index=False)


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


if __name__ == '__main__':

    data_creator = DataCreator(create_unlabeled=False)