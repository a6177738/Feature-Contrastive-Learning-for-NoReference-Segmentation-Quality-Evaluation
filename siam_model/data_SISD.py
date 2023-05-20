import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random

from data_aug import sim

class Data(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self._imgpath = os.path.join('%s', 'original', '%s.jpg')
        #self._midlinepath = os.path.join('%s', 'mid_line', '%s.png')
        self._seg = os.path.join('%s', 'anchor', '%s.png')
        self._pos = os.path.join('%s', 'positive', '%s.png')
        self._neg = os.path.join('%s', 'negative', '%s.png')
        self._noise = os.path.join('%s', 'noise', '%s.png')
        self._othergt = os.path.join('%s', 'othergt', '%s.png')
        self.ids = list()
        for line in open(os.path.join(root, 'img_name.txt')):
            self.ids.append((root, line.replace('.jpg','').strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        gt = cv2.imread(self._seg % img_id)
        print(img_id[1])


        # 其中neg是筛选的neg样本，across是跨图搜索标签作为负样本，cut是pos样本裁剪大小
        noise_neg_across = random.randint(0,1)
        #cut为裁剪比例，剩余作为负样本
        pos = gt
        # if noise_neg_across==0:
        #     neg = cv2.imread((self._noise % img_id))

        # if noise_neg_across==1:
        #neg = cv2.imread(self._neg % img_id)
        # else:
        id = random.randint(0, len(self.ids)-1)
        neg = cv2.imread(self._seg % self.ids[id])

        img_resize = cv2.resize(img,(320,320))
        pos = cv2.resize(pos,(320,320))
        neg = cv2.resize(neg,(320,320))
        gt = cv2.resize(gt,(320,320))
        pos_label = sim(pos,gt) #(321,321) value:(255 or 0)
        neg_label = sim(neg,gt)
        img_resize = img_resize.transpose(2,0,1)
        pos = pos.transpose(2, 0, 1)
        neg = neg.transpose(2, 0, 1)
        #用于相似性匹配
        pos_img = np.concatenate((img_resize,pos),0)
        neg_img = np.concatenate((img_resize,neg),0)
        return (pos_img, neg_img, pos_label, neg_label)
    def __len__(self):
        return len(self.ids)