# -*- coding: utf-8 -*-
""" 
@Time    : 2022/1/21 10:50
@Author  : Johnson
@FileName: detect.py
"""


def detect():
    source = '../input/global-wheat-detection/test/'
    weights = 'weights/best.pt'
    if not os.path.exists(weights):
        weights = '../input/yolov5/bestv4.pt'
    imgsz = 1024
    conf_thres = 0.5
    iou_thres = 0.6
    is_TTA = True

    imagenames = os.listdir(source)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    dataset = LoadImages(source, img_size=imgsz)

    results = []
    fig, ax = plt.subplots(5, 2, figsize=(30, 70))
    count = 0
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # for path, img, im0s, _ in dataset:
    for name in imagenames:
        image_id = name.split('.')[0]
        im01 = cv2.imread('%s/%s.jpg' % (source, image_id))  # BGR
        assert im01 is not None, 'Image Not Found '
        # Padded resize
        im_w, im_h = im01.shape[:2]
        if is_TTA:
            enboxes = []
            enscores = []
            for i in range(4):
                im0 = TTAImage(im01, i)
                boxes, scores = detect1Image(im0, imgsz, model, device, conf_thres, iou_thres)
                for _ in range(3 - i):
                    boxes = rotBoxes90(boxes, im_w, im_h)

                if 1:  # i<3:
                    enboxes.append(boxes)
                    enscores.append(scores)
            boxes, scores = detect1Image(im01, imgsz, model, device, conf_thres, iou_thres)
            enboxes.append(boxes)
            enscores.append(scores)

            boxes, scores, labels = run_wbf(enboxes, enscores, image_size=im_w, iou_thr=0.6, skip_box_thr=0.5)
            boxes = boxes.astype(np.int32).clip(min=0, max=im_w)
        else:
            boxes, scores = detect1Image(im01, imgsz, model, device, conf_thres, iou_thres)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >= float(0.05)]
        if count < 10:
            # sample = image.permute(1,2,0).cpu().numpy()
            for box, score in zip(boxes, scores):
                cv2.rectangle(im0,
                              (box[0], box[1]),
                              (box[2] + box[0], box[3] + box[1]),
                              (220, 0, 0), 2)
                cv2.putText(im0, '%.2f' % (score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2, cv2.LINE_AA)
            ax[count % 5][count // 5].imshow(im0)
            count += 1

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)
    return results