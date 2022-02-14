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

# Which IOU level to evaluate (Competition metric tests 0.3 to 0.8 with step of 0.05)
eval_IOU = 0.65

# Confidence scores of true positives, false positives and count false negatives
TP = []  # Confidence scores of true positives
FP = []  # Confidence scores of true positives
FN = 0  # Count of false negative boxes

for i in tqdm(range(len(image_paths))):
    TEST_IMAGE_PATH = image_paths[i]
    img = cv2.imread(TEST_IMAGE_PATH)
    img = cv2.imread(TEST_IMAGE_PATH)[..., ::-1]
    bboxes, scores = predict(model, img, size=IMG_SIZE, augment=AUGMENT)

    # Test YOLOV5
    gt0 = gt[i]
    if len(bboxes) == 0:
        # all gt are false negative
        FN += len(gt0)
    else:
        bb = bboxes.copy().tolist()
        for idx, b in enumerate(bb):
            b.append(scores[idx])
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