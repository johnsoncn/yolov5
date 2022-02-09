
import os
from tqdm import tqdm
pseudo_root = '/home/dingchaofan/dataset/starfish/data/pseudo_labels_2/train'

# pseudo_label_dir = os.listdir(root_dir)
#
# print(pseudo_label_dir)
#
# for dirpath, dirnames, filenames in os.walk('/home/dingchaofan/dataset/starfish/data/pseudo_labels_2/train'):
#     print(dirpath)
#     print(dirnames)
#     print(filenames)

labels = []

for root, dirs, files in os.walk(pseudo_root, topdown=True):


    for name in files:
        # print(os.path.join(root, name))
        labels.append(os.path.join(root, name))

print(labels)
print(len(labels))
    # for name in dirs:
    #     print(os.path.join(root, name))

    # print(root)