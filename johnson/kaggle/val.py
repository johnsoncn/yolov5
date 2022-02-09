# # -*- coding: utf-8 -*-
# """
# @Time    : 2022/1/21 10:23
# @Author  : Johnson
# @FileName: val.py
# """
#
#
# CV=True
#
# if CV:
#     for i in range(3600,10000,1200):
#
#         print("#######################################\n"*3, f'Starting Inference for image size {i}')
#         start_time = time.time()
#         #
#         # !python val.py --data ./fold0.yaml\
#         #     --weights /kaggle/input/reef-baseline-fold12/l6_3600_uflip_vm5_f12_up/f1/best.pt\
#         #     --imgsz $i\
#         #     --batch 4\
#         #     --conf-thres 0.01\
#         #     --iou-thres 0.3\
#         #     --save-txt\
#         #     --save-conf\
#         #     --exist-ok
#         t=(time.time() - start_time)/60
#         print(f'Inference Complete in {t:.3f} minutes')
#         print('Starting Cross Validation')
#         start_time = time.time()
#         scores = []
#         for j in range(15,40):
#             confidence=j/100
#             gt_bboxs_list, prd_bboxs_list = [], []
#
#             count=0
#             for image_file in paths:
#                 gt_bboxs = []; prd_bboxs = []
#                 with open(image_file, 'r') as f:
#                     while True:
#                         r = f.readline().rstrip()
#                         if not r: break
#                         r = r.split()[1:]
#                         bbox = np.array(list(map(float, r))); gt_bboxs.append(bbox)
#
#                 pred_path = '/kaggle/working/yolov5/runs/val/exp/labels/'
#                 pred_file = pred_path+image_file[27:]
#
#                 no_anns = True
#                 if os.path.exists(pred_file):
#                     with open(pred_file, 'r') as f:
#                         while True:
#                             r = f.readline().rstrip()
#                             if not r: break
#                             r = r.split()[1:]; r = [r[4], *r[:4]]
#                             conf=float(r[0])
#                             if conf>confidence:
#                                 bbox = np.array(list(map(float, r)))
#                                 prd_bboxs.append(bbox)
#                                 no_anns = False
#
#                 if no_anns: count+=1
#
#                 gt_bboxs, prd_bboxs= np.array(gt_bboxs), np.array(prd_bboxs)
#                 prd_bboxs_list.append(prd_bboxs); gt_bboxs_list.append(gt_bboxs)
#
#             score = calc_f2_score(gt_bboxs_list, prd_bboxs_list, verbose=False)
#             scores.append([score, confidence, count])
#             if confidence%5: print(f'confidence: {confidence}, images w/o anns: {count}, total: {val_len}')
#
#         best = max(scores)
#         print(f'best confidence: {best[1]}, images w/o anns: {best[2]}, total: {val_len}')
#         print(f'img size: {i}, f2 score: {best[0]}')
#         t=(time.time() - start_time)/60
#         print(f'cross validation complete in {t.3f} minutes')
#         torch.cuda.empty_cache()