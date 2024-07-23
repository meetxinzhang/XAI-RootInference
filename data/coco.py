import os
import random
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torchvision import transforms

random.seed(0)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
resize_func = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

normal_func = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

mask_trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
])

cat_id_coco = {
    5: 'airplane',
    2: 'bicycle',
    16: 'bird',
    9: 'boat',
    44: 'bottle',
    6: 'bus',
    3: 'car',
    17: 'cat',
    62: 'chair',
    21: 'cow',
    67: 'dining table',
    18: 'dog',
    19: 'horse',
    4: 'motorcycle',
    1: 'person',
    64: 'potted plant',
    20: 'sheep',
    63: 'couch',
    7: 'train',
    72: 'tv'
}

label_my = {
    5:  0,
    2:  1,
    16: 2,
    9:  3,
    44: 4,
    6:  5,
    3:  6,
    17: 7,
    62: 8,
    21: 9,
    67: 10,
    18: 11,
    19: 12,
    4:  13,
    1:  14,
    64: 15,
    20: 16,
    63: 17,
    7:  18,
    72: 19
}


class COCOSegDataset(data.Dataset):
    def __init__(self, first_classes=True):
        super(COCOSegDataset, self).__init__()
        json_path = "/coco/annotations/instances_val2017.json"
        self.img_path = "/coco/val2017"

        # load coco data
        self.coco = COCO(annotation_file=json_path)
        # get all image index info
        # self.ids = list(sorted(self.coco.imgs.keys()))
        # print("number of images: {}".format(len(self.ids)))
        # print(self.ids)

        # select 20 classes of VOC
        self.img_ids_voc = []
        self.img_label_voc = []
        for i, k in enumerate(cat_id_coco.keys()):
            ids = self.coco.getImgIds(catIds=k)
            labels = [i]*len(ids)

            self.img_ids_voc.extend(ids)
            self.img_label_voc.extend(labels)

        # get all coco class labels
        # self.coco_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        # print(self.coco_classes)
        self.first_classes = first_classes

    def __getitem__(self, index):
        img_id = self.img_ids_voc[index]
        label = self.img_label_voc[index]
        # 获取对应图像id的所有annotations idx信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # 根据annotations idx信息获取所有标注信息
        targets = self.coco.loadAnns(ann_ids)

        # get image file name
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # read image
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        img_w, img_h = img.size

        # if self.first_classes:
        try:
            target = targets[0]
        except IndexError:
            return None, None, None
        # label = target["category_id"]
        polygons = target["segmentation"]  # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(axis=2)
        heatmap = Image.fromarray(mask.astype(np.uint8))
        # else:
        # masks = []
        # cats = []
        # for target in targets:
        #     cats.append(target["category_id"])  # get object class id
        #     polygons = target["segmentation"]  # get object polygons
        #     rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        #     mask = coco_mask.decode(rles)
        #     if len(mask.shape) < 3:
        #         mask = mask[..., None]
        #     mask = mask.any(axis=2)
        #     masks.append(mask)
        #
        # cats = np.array(cats, dtype=np.int32)
        # if masks:
        #     masks = np.stack(masks, axis=0)
        # else:
        #     masks = np.zeros((0, img_h, img_w), dtype=np.uint8)
        #
        # # merge all instance masks into a single segmentation map
        # # with its corresponding categories
        # heatmap = (masks * cats[:, None, None]).max(axis=0)
        # # discard overlapping instances
        # heatmap[masks.sum(0) > 1] = 255
        # heatmap = Image.fromarray(heatmap.astype(np.uint8))
        # # heatmap.putpalette(pallette)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        re_img = resize_func(img)
        img_tensor = normal_func(re_img)

        heatmap = np.array(mask_trans(heatmap)).astype('int32')
        heatmap[heatmap == 255] = -1
        heatmap = torch.from_numpy(heatmap).long()
        heatmap = torch.where(heatmap > 0, 1, 0)

        return img_tensor, heatmap, label

    def __len__(self):
        return len(self.img_ids_voc)

# json_path = "/Datasets/coco/annotations/instances_val2017.json"
# img_path = "/Datasets/coco/val2017"
#
# # random pallette
# pallette = [0, 0, 0] + [random.randint(0, 255) for _ in range(255*3)]
#
# # load coco data
# coco = COCO(annotation_file=json_path)
#
# # get all image index info
# ids = list(sorted(coco.imgs.keys()))
# print("number of images: {}".format(len(ids)))
#
# # get all coco class labels
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
# print(coco_classes)
#
# # 遍历前三张图像
# classsss = []
# for img_id in ids[:1000]:
#     # 获取对应图像id的所有annotations idx信息
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     # 根据annotations idx信息获取所有标注信息
#     targets = coco.loadAnns(ann_ids)
#
#     # print('aaaaaaa', img_id)
#     # get image file name
#     path = coco.loadImgs(img_id)[0]['file_name']
#     # read image
#     img = Image.open(os.path.join(img_path, path)).convert('RGB')
#     img_w, img_h = img.size
#
#     # masks = []
#     # cats = []
#     # for target in targets:
#     #     cats.append(target["category_id"])  # get object class id
#     #     polygons = target["segmentation"]   # get object polygons
#     #     rles = coco_mask.frPyObjects(polygons, img_h, img_w)
#     #     mask = coco_mask.decode(rles)
#     #     if len(mask.shape) < 3:
#     #         mask = mask[..., None]
#     #     mask = mask.any(axis=2)
#     #     masks.append(mask)
#     #
#     # cats = np.array(cats, dtype=np.int32)
#     # if masks:
#     #     masks = np.stack(masks, axis=0)
#     # else:
#     #     masks = np.zeros((0, img_h, img_w), dtype=np.uint8)
#     #
#     # # merge all instance masks into a single segmentation map
#     # # with its corresponding categories
#     # heatmap = (masks * cats[:, None, None]).max(axis=0)
#     # # discard overlapping instances
#     # heatmap[masks.sum(0) > 1] = 255
#     # heatmap = Image.fromarray(heatmap.astype(np.uint8))
#     # # heatmap.putpalette
#     try:
#         target = targets[0]
#     except IndexError:
#         continue
#     label = target["category_id"]
#     print(label)
#     classsss.append(label)
#     polygons = target["segmentation"]  # get object polygons
#     rles = coco_mask.frPyObjects(polygons, img_h, img_w)
#     mask = coco_mask.decode(rles)
#     if len(mask.shape) < 3:
#         mask = mask[..., None]
#     mask = mask.any(axis=2)
#     heatmap = Image.fromarray(mask.astype(np.uint8))
#
#     heatmap = np.array(mask_trans(heatmap)).astype('int32')
#     heatmap[heatmap == 255] = -1
#     heatmap = torch.from_numpy(heatmap).long()
#     heatmap = torch.where(heatmap > 0, 1, 0)
#
#     # plt.imshow(heatmap)
#     # plt.show()
# print(list(set(classsss)))
