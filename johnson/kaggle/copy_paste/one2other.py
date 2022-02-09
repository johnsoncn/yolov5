

import csv
from itertools import islice
import cv2


TRAIN_PATH = '/home/dingchaofan/dataset/starfish/data'

video_id = 0
sequence = 1
video_frame = 2
sequence_frame = 3
image_id = 4
annotations = 5


def get_path(row):
    image_path = f'{TRAIN_PATH}/train_images/video_{row[video_id]}/{row[video_frame]}.jpg'
    return image_path

def coco2voc(box):

    xmin = box[0]
    ymin = box[1]
    xmax = box[0] + box[2]
    ymax = box[1] + box[3]
    box = (int(xmin), int(ymin), int(xmax), int(ymax))
    return box

def draw(image, pos):

    for i in range(len(pos)):
        box = coco2voc(pos[i])
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    return image


def data_loader():
    with_label = 0
    no_label = 0

    with open(TRAIN_PATH + '/train.csv') as csv_file:
        csv_reader = csv.reader(csv_file) # delimiter=','
        for row in islice(csv_reader, 1, None):
            image_path = get_path(row)
            labels = eval(row[annotations])
            if labels:
                labels_list = []
                for label in labels:
                    x, y, w, h = label['x'], label['y'], label['width'], label['height']
                    labels_list.append([x,y,w,h])
                _dict = {image_path:labels_list}
                with_label += 1
            else:
                print(image_path)
                no_label += 1

    print(with_label)
    print(no_label)


def copy_and_paste(src_img, dst_img, annot):
    import os

    test_dir = 'test_dataset/7/'
    os.makedirs(test_dir, exist_ok=True)

    src_img_read = cv2.imread(src_img)
    cv2.imwrite(test_dir+ "src_img.jpg", src_img_read)
    src_img_withlabel = draw(src_img_read.copy(), annot)
    cv2.imwrite(test_dir+ "src_img_withlabel.jpg", src_img_withlabel)

    dst_img_read = cv2.imread(dst_img)
    cv2.imwrite(test_dir+ "dst_img_1.jpg", dst_img_read)

    for i,an in enumerate(annot):
        an = coco2voc(an)
        dst_img_read[an[1]:an[3], an[0]:an[2],:] = src_img_read.copy()[an[1]:an[3], an[0]:an[2],:]
        cv2.imwrite(test_dir + "i.jpg", src_img_read.copy()[an[1]:an[3], an[0]:an[2]])
    cv2.imwrite(test_dir+ "dst_img_2.jpg", dst_img_read)
    dst_img_withlabel = draw(dst_img_read, annot)
    cv2.imwrite(test_dir+ "dst_img_withlabel.jpg", dst_img_withlabel)

    return dst_img, annot


if __name__ == "__main__":

    """
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/8863.jpg': [[457, 556, 57, 39]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/7965.jpg': [[512, 257, 35, 20]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/6367.jpg': [[172, 134, 33, 28]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/6252.jpg': [[513, 206, 36, 37], [490, 226, 33, 31], [465, 249, 37, 35]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/5924.jpg': [[526, 643, 51, 50], [504, 680, 36, 31], [303, 339, 26, 27], [324, 351, 29, 29]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/5914.jpg': [[554, 560, 44, 46], [556, 609, 39, 38], [533, 592, 36, 33], [336, 300, 25, 26], [355, 308, 29, 29]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/5898.jpg': [[757, 663, 30, 26], [590, 411, 39, 36], [590, 448, 34, 35], [568, 434, 33, 31], [376, 208, 24, 24], [396, 213, 29, 29]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/5854.jpg': [[783, 358, 50, 39], [678, 280, 21, 26], [660, 272, 26, 25]]}
{'/home/dingchaofan/dataset/starfish/data/train_images/video_2/5828.jpg': [[968, 289, 67, 54], [1161, 354, 75, 71], [215, 591, 72, 60], [723, 0, 50, 13], [181, 546, 89, 77], [448, 415, 62, 67], [93, 558, 63, 48], [349, 602, 58, 63], [384, 434, 31, 31]]}

/home/dingchaofan/dataset/starfish/data/train_images/video_2/10745.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10728.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10681.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10530.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10457.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10387.jpg
/home/dingchaofan/dataset/starfish/data/train_images/video_2/10350.jpg

    """

    src_img = '/home/dingchaofan/dataset/starfish/data/train_images/video_2/5924.jpg'
    annot = [[526, 643, 51, 50], [504, 680, 36, 31], [303, 339, 26, 27], [324, 351, 29, 29]]

    dst_img = '/home/dingchaofan/dataset/starfish/data/train_images/video_2/10530.jpg'
    copy_and_paste(src_img, dst_img, annot)

    # data_loader()