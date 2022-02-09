import cv2
import os


class VisualLabel(object):

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.img_files = os.listdir(self.image_dir)
        self.img_files.sort()

        self.label_files = os.listdir(self.label_dir)
        self.label_files.sort()

    def read_img(self, image_path):
        image = cv2.imread(image_path)
        return image

    def read_txt(self, txt_path):
        pos = []
        with open(txt_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                p_tmp = [float(i) for i in lines.split(' ')]  # 将整行数据分割处理
                pos.append(p_tmp)
        return pos

    def yolo2coco(self, size, box):
        xmin = (box[1] - box[3] / 2.) * size[1]
        xmax = (box[1] + box[3] / 2.) * size[1]
        ymin = (box[2] - box[4] / 2.) * size[0]
        ymax = (box[2] + box[4] / 2.) * size[0]
        box = (int(xmin), int(ymin), int(xmax), int(ymax))
        return box

    def draw(self, image, pos):
        tl = int((image.shape[0] + image.shape[1]) / 2) + 1
        lf = max(tl - 1, 1)
        for i in range(len(pos)):
            label = str(int(pos[i][0]))
            box = self.yolo2coco(image.shape, pos[i])
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            cv2.putText(image, label, (box[0], box[1] - 2), 0, tl, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        return image

    def main(self, save=False):
        for name in self.img_files:
            image = self.read_img(os.path.join(self.image_dir, name))
            pos = self.read_txt(os.path.join(self.label_dir, name.replace('png', 'txt')))
            drawed_image = self.draw(image=image, pos=pos)

            if save:
                save_to = 'ui_visual_label/' + name
                cv2.imwrite(save_to, drawed_image)
                print('save to :', save_to)


if __name__ == "__main__":
    image_dir = "/home/dingchaofan/yolov5/data/dataset_1207/dataset_1207/images"
    label_dir = "/home/dingchaofan/yolov5/data/dataset_1207/dataset_1207/labels"

    visual = VisualLabel(image_dir, label_dir)
    visual.main(save=False)