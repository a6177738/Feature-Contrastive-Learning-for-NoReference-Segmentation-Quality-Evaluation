import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random

from data_aug import sim

class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self._imgpath = os.path.join('%s', 'original', '%s.jpg')
        self._seg = os.path.join('%s', 'anchor', '%s.png')
        self._pos = os.path.join('%s', 'positive', '%s.png')
        self._neg = os.path.join('%s', 'negative', '%s.png')
        self._cam = os.path.join('%s', 'cam', '%s.png')  #
        self.ids = list()
        txt = 'img_name.txt'
        for line in open(os.path.join(root, txt)):
            self.ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        gt = cv2.imread(self._seg % img_id)
        cam = cv2.imread(self._cam % img_id, 0)
        # 其中neg是筛选的neg样本，across是跨图搜索标签作为负样本，cut是pos样本裁剪大小
        gt_neg1_neg2_across = random.randint(0,3)
        if gt_neg1_neg2_across==0:
           prelabel = cv2.imread(self._seg % img_id)
        elif gt_neg1_neg2_across==1:
            prelabel = cv2.imread(self._neg % img_id)
        elif gt_neg1_neg2_across==2:
            prelabel = cv2.imread(self._pos % img_id)
        else:
            id = random.randint(0, len(self.ids) - 1)
            prelabel = cv2.imread(self._seg % self.ids[id])

        img = cv2.resize(img,(320,320))
        gt = cv2.resize(gt,(320,320))
        prelabel = cv2.resize(prelabel,(320,320))
        cam = cv2.resize(cam,(320,320))
        label = sim(prelabel,gt) #(320,320) value:(255 or 0)
        img_resize = img.transpose(2,0,1)
        prelabel = prelabel.transpose(2, 0, 1)
        #用于相似性匹配
        con_img = np.concatenate((img_resize,prelabel),0)

        return (con_img, label, cam, img_id[1])
    def __len__(self):
        return len(self.ids)