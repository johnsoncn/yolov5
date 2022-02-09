# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/24 11:42
@Author  : Johnson
@FileName: tracking.py
"""

import numpy as np
from norfair import Detection, Tracker
import cv2

# Helper to convert bbox in format [x_min, y_min, x_max, y_max, score] to norfair.Detection class
def to_norfair(detects, frame_id):
    result = []
    for x_min, y_min, x_max, y_max, score in detects:
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        result.append(Detection(points=np.array([xc, yc]), scores=np.array([score]), data=np.array([w, h, frame_id])))

    return result


# Euclidean distance function to match detections on this frame with tracked_objects from previous frames
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def tracking_function(tracker, frame_id, bboxes, scores):
    detects = []
    predictions = []

    if len(scores) > 0:
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = scores[i]
            x_min = int(box[0])
            y_min = int(box[1])
            bbox_width = int(box[2])
            bbox_height = int(box[3])
            detects.append([x_min, y_min, x_min + bbox_width, y_min + bbox_height, score])
            predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))
    #             print(predictions[:-1])
    # Update tracks using detects from current frame
    tracked_objects = tracker.update(detections=to_norfair(detects, frame_id))
    for tobj in tracked_objects:
        bbox_width, bbox_height, last_detected_frame_id = tobj.last_detection.data
        if last_detected_frame_id == frame_id:  # Skip objects that were detected on current frame
            continue
        # Add objects that have no detections on current frame to predictions
        xc, yc = tobj.estimate[0]
        x_min, y_min = int(round(xc - bbox_width / 2)), int(round(yc - bbox_height / 2))
        score = tobj.last_detection.scores[0]

        predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))

    return predictions


tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=30,
    hit_inertia_min=3,
    hit_inertia_max=6,
    initialization_delay=1,
)

# model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
#
# frame_id = 0
# for idx, (img, pred_df) in enumerate(tqdm(iter_test)):
#     # if FDA_aug:
#     #     img = FDA_trans(image=img)['image']
#     bboxes, confs = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
#
#     predictions = tracking_function(tracker, frame_id, bboxes, confs)
#     prediction_str = ' '.join(predictions)
#     pred_df['annotations'] = prediction_str
#
#     env.predict(pred_df)
#     # if frame_id < 3:
#     #     if len(predict_box) > 0:
#     #         box = [list(map(int, box.split(' ')[1:])) for box in predictions]
#     #     else:
#     #         box = []
#     #     display(show_img(img, box, bbox_format='coco'))
#     #     print('Prediction:', pred_df)
#     frame_id += 1
