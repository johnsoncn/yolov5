# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/6 17:47
@Author  : Johnson
@FileName: test.py
"""

import os

root = '/home/dingchaofan/dataset/starfish/data/fulldata/images/valid'
roota = '/home/dingchaofan/dataset/starfish/data/fulldata/labels/valid'

print(len(os.listdir(root)))
print(len(os.listdir(roota)))


# import torch
#
# # Model
# def load_model(Best_Model, conf=0.25, iou=0.50):
#     model = torch.hub.load('/home/dingchaofan/yolov5',  # /kaggle/input/barrie-reef-yolo5/yolov5
#                            'custom',
#                            path=Best_Model,
#                            source='local',
#                            force_reload=True)  # local repo
#     model.conf = conf  # NMS confidence threshold
#     model.iou = iou  # NMS IoU threshold
#     model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
#     model.multi_label = False  # NMS multiple labels per box
#     model.max_det = 1000  # maximum number of detections per image
#
#     return model
#
# Best_Model = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt'
#
# # Image
# img = 'https://ultralytics.com/images/zidane.jpg'
#
# # Inference
# model = load_model(Best_Model = '/home/dingchaofan/yolov5/runs/train/exp34/weights/last.pt')
# results = model(img)
# print(results)
#
# # results.pandas().xyxy[0]

# import greatbarrierreef
# env = greatbarrierreef.make_env()# initialize the environment
# iter_test = env.iter_test()      # an iterator which loops over the test set and sample submission
#
# for item in iter_test:
#     print(item[0])
#     print('---')
#     print(item[1])
#     break

# import greatbarrierreef
# env = greatbarrierreef.make_env()   # initialize the environment
# iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission
# for (pixel_array, sample_prediction_df) in iter_test:
#     sample_prediction_df['annotations'] = '0.5 0 0 100 100'  # make your predictions here
#     env.predict(sample_prediction_df)   # register your predictions


