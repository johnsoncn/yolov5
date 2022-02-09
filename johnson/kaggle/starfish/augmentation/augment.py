
import pandas as pd
import cv2
import os
import albumentations as A
import glob
import numpy as np
import matplotlib.pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar
# from more_itertools import chunked
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
from tqdm.notebook import tqdm
tqdm.pandas()
import ast

class CFG:
    path_original = "../input/tensorflow-great-barrier-reef/train.csv"
    fold_index = 0
    path_dataset = "./aug"
    os.makedirs(path_dataset, exist_ok=True)
    visualize = False
    worker=4
    batch_size = 128
    kfold = True
    aug_box = True
    aug_box_time = 8 ### total times augmentation
    use_coco2yolo = False

#!rm -rf $CFG.path_dataset

def get_path(row):
    row['image_path'] = f'../input/tensorflow-great-barrier-reef/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    return row


if CFG.kfold:
    from sklearn.model_selection import GroupKFold

    df = pd.read_csv(CFG.path_original)
    df = df.progress_apply(get_path, axis=1)
    FDA_image = df[df['annotations'] == '[]']['image_path'].tolist()
    df = df[df['annotations'] != '[]']
    df['annotations'] = df['annotations'].progress_apply(lambda x: ast.literal_eval(x))

    kf = GroupKFold(n_splits=5)
    df = df.reset_index(drop=True)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y=df.video_id.tolist(), groups=df.sequence)):
        df.loc[val_idx, 'fold'] = fold
    display(df.fold.value_counts())
train = df[df['fold'] != CFG.fold_index]
valid = df[df['fold'] == CFG.fold_index]

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

train['bboxes'] = train.annotations.progress_apply(get_bbox)
valid['bboxes'] = valid.annotations.progress_apply(get_bbox)


for folder_type in ['images', 'labels']:
    path_phase = CFG.path_dataset + '/' + folder_type
    os.makedirs(path_phase, exist_ok=True)
    for phase in ['train', 'valid']:
        path_type = path_phase + '/' + phase
        os.makedirs(path_type, exist_ok=True)


### aug image ###
def write_box_into_image(bouding_box, path_image):
    ### bouding_box, path_image
    image = cv2.imread(path_image)
    #     print(image.shape)
    for bb in bouding_box:
        x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
        print(bb)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image


class HE_HSV(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Histogram equalisation on the V-channel
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

        # convert image back from HSV to RGB
        image_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return image_hsv


class RecoverHE(A.ImageOnlyTransform):
    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, sceneRadiance, **params):
        for i in range(3):
            sceneRadiance[:, :, i] = cv2.equalizeHist(sceneRadiance[:, :, i])
        return sceneRadiance


class CLAHE_HSV(A.ImageOnlyTransform):

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(20, 20))
        v = clahe.apply(v)

        hsv_img = np.dstack((h, s, v))

        rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        return rgb


class RecoverCLAHE(A.ImageOnlyTransform):

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, sceneRadiance, **params):
        clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
        for i in range(3):
            sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))

        return sceneRadiance


class Gamma_enhancement(A.ImageOnlyTransform):

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)
        self.gamma = 1 / 0.6
        self.R = 255.0

    def apply(self, image, **params):
        return (self.R * np.power(image.astype(np.uint32) / self.R, self.gamma)).astype(np.uint8)


class RecoverGC(A.ImageOnlyTransform):

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, sceneRadiance, **params):
        sceneRadiance = sceneRadiance / 255.0
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
        for i in range(3):
            sceneRadiance[:, :, i] = np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 3.2)
        sceneRadiance = np.clip(sceneRadiance * 255, 0, 255)
        sceneRadiance = np.uint8(sceneRadiance)
        return sceneRadiance


class RecoverICM(A.ImageOnlyTransform):

    def __init__(self, p: float = 0.5, always_apply=False):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        img_stre = stretching(iamge)
        sceneRadiance = sceneRadianceRGB(img_stre)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)

        return sceneRadiance

def show_one_image(image):
    plt.figure(figsize=(20, 15))
    plt.gcf().set_dpi(100)
    plt.imshow(image)
    plt.show()

# train = cross_validation[cross_validation['fold']!=CFG.fold_index]
# valid = cross_validation[cross_validation['fold']==CFG.fold_index]

def bbox_to_txt(bboxes):
    """
    Convert a list of bbox into a string in YOLO format (to write a file).
    @bboxes : numpy array of bounding boxes
    return : a string for each object in new line: <object-class> <x> <y> <width> <height>
    """
    txt=''
    for index,l in enumerate(bboxes):
        l = [str(x) for x in l[:4]]
        l = ' '.join(l)
        txt +=  '0' +' ' + l + '\n'
    return txt

def prepare_data(df, total_aug=10):
    ### return aug df
    df_aug = pd.DataFrame(np.repeat(df.values, total_aug, axis=0), columns=df.columns)
    df_aug['aug_index'] = np.repeat(np.arange(1,total_aug+1).reshape(1,-1),df.shape[0], axis=0).reshape(-1)
    df_aug = df_aug.sample(frac=1)
    df_aug = df_aug.reset_index(drop=True)
    return df_aug
train = prepare_data(train, CFG.aug_box_time)
valid = prepare_data(valid,1)


class AUG_DATASET(Dataset):

    def __init__(self, df, mode, transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def coco2yolo(self, bboxes, image_height=720, image_width=1280):
        """
        coco => [xmin, ymin, w, h]
        yolo => [xmid, ymid, w, h] (normalized)
        """
        bboxes = np.array(bboxes)
        bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

        # normolizinig
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] / image_width
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] / image_height

        # converstion (xmin, ymin) => (xmid, ymid)
        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2

        return bboxes

    def coord_to_box(self, bouding_box, image):
        box_yolo_format = []
        height, width = image.shape[0], image.shape[1]

        if CFG.use_coco2yolo:
            box_yolo_format = self.coco2yolo(bouding_box)
            box_yolo_format = np.clip(box_yolo_format, 0, 1)
            label = np.repeat([0], box_yolo_format.shape[0]).reshape(-1, 1)
            box_yolo_format = np.append(box_yolo_format, label, axis=1)
        else:
            for bb in bouding_box:
                label = [max(0, bb[0]), max(0, bb[1]), min(bb[0] + bb[2], 1280), min(720, bb[1] + bb[3]), '0']
                bbox_albu = A.convert_bbox_to_albumentations(label, source_format='pascal_voc', rows=height, cols=width)
                bbox_yolo = A.convert_bbox_from_albumentations(bbox_albu, target_format='yolo', rows=height, cols=width,
                                                               check_validity=True)
                clip_box = [np.clip(value, 0, 1) for value in bbox_yolo[:-1]] + [bbox_yolo[-1]]
                box_yolo_format.append(clip_box)
        return box_yolo_format

    def bbox_to_txt(self, bboxes):
        """
        Convert a list of bbox into a string in YOLO format (to write a file).
        @bboxes : numpy array of bounding boxes
        return : a string for each object in new line: <object-class> <x> <y> <width> <height>
        """
        txt = ''
        for index, l in enumerate(bboxes):
            l = [str(x) for x in l[:4]]
            l = ' '.join(l)
            txt += '0' + ' ' + l + '\n'
        return txt

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['image_path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_index = row['aug_index']
        list_info = path.split('/')
        image_name = list_info[-2] + '_' + list_info[-1].split('.')[0]
        box = row['bboxes']
        bounding_box = self.coord_to_box(box, img)

        if self.transform is not None and self.mode == 'train':
            res = self.transform(image=img, bboxes=bounding_box)
            img = res['image']
            bounding_box = res['bboxes']

        box_yolo_format = self.bbox_to_txt(bounding_box)
        return img, box_yolo_format, image_name, aug_index


def get_transforms(phase):
    if not CFG.aug_box:
        return None
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.3),
            A.OneOf([
                #                     A.FDA(reference_images=FDA_image,p=0.5),
                HE_HSV(0.75),
                CLAHE_HSV(0.75),
                Gamma_enhancement(0.75)
            ], p=0.75),
            #                 A.ShiftScaleRotate(scale_limit = 0, rotate_limit=30, p=0.3, border_mode=0)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, min_area=500))
    else:
        return None
#         return A.Compose([])

dataset = {
    phase: AUG_DATASET(eval(phase), mode=phase, transform = get_transforms(phase=phase)) for phase in ['train','valid']
}

dataloader = {
    phase: Data.DataLoader(dataset=dataset[phase], num_workers=CFG.worker, batch_size=CFG.batch_size, shuffle=False, drop_last=False,\
                           pin_memory = False) for phase in ['train','valid']
}
for phase in ['train','valid']:
    for aug_img, aug_box, image_name, aug_index in progress_bar(dataloader[phase]):
        for idx, image in enumerate(aug_img):
            name = image_name[idx]
            aug_index_name = aug_index[idx]
            new_name =  "{}_{}".format(name,aug_index_name)
            image = aug_img[idx]
            box = aug_box[idx]

            path_txt = CFG.path_dataset + "/" + "labels" + "/" + phase + "/" + new_name + ".txt"
            path_jpg = CFG.path_dataset + "/" + "images" + "/" + phase + "/" + new_name + ".jpg"
            is_path = os.path.exists(path_jpg)
            image = image.numpy()
            cv2.imwrite(path_jpg, image[...,::-1])
            txt_file = open(path_txt, "w")
            txt_file.write(box)
            txt_file.close()
        break
    break