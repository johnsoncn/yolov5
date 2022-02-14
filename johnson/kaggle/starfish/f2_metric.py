# -*- coding: utf-8 -*-
""" 
@Time    : 2022/2/14 16:29
@Author  : Johnson
@FileName: f2_score.py
"""
from tqdm import tqdm
import cv2
image_paths = ''
# gt =


def IOU_coco(bbox1, bbox2):
    '''
        adapted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    '''
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = bbox1[2] * bbox1[3]
    bb2_area = bbox2[2] * bbox2[3]
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def f2(preds, gts):
    # Which IOU level to evaluate (Competition metric tests 0.3 to 0.8 with step of 0.05)
    eval_IOU = 0.65

    # Confidence scores of true positives, false positives and count false negatives
    TP = []  # Confidence scores of true positives
    FP = []  # Confidence scores of true positives
    FN = 0  # Count of false negative boxes

    # Test YOLOV5
    gt0 = gts
    if len(preds) == 0:
        # all gt are false negative
        FN += len(gt0)
    else:
        bb = preds.copy().tolist()
        for idx, b in enumerate(bb):
            b.append(preds[idx])
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

    return TP, FP, FN

if __name__ == "__main__":

    # pred, gt = [], []
    TP, FP, FN = f2(pred, gt)

    print(TP, FP, FN)