# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/26 14:29
@Author  : Johnson
@FileName: mixup_background.py
"""
import os
import random
import numpy as np
import cv2

class Mixup_backup(object):
    """Mixup images & bbox
    Args:
        prob (float): the probability of carrying out mixup process.
        lambd (float): the parameter for mixup.
        mixup (bool): mixup switch.
        json_path (string): the path to dataset json file.
    """

    def __init__(self, p=0.5, lambd=0.5, backup_path='no_object_pictures', to_float=False):
        self.lambd = lambd
        self.prob = p
        self.backup_path = backup_path
        self.backup_imgs = os.listdir(self.backup_path)
        self.backup_length = len(self.backup_imgs)
        self.to_float = to_float

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            img1 = results['img']
            labels1 = results['gt_labels']

            self.lambd = 0.5
            idx = np.random.randint(0,self.backup_length)
            img2 = cv2.imread(os.path.join(self.backup_path,self.backup_imgs[idx]))
            height = max(img1.shape[0], img2.shape[0])
            width = max(img1.shape[1], img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
            mixup_image[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - self.lambd)
            if not self.to_float:
                mixup_image = mixup_image.astype('uint8')

            results['img'] = mixup_image
            return results
        else:
            return results