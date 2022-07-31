import numpy as np
import copy
import cv2
import random
from imgaug import augmenters as iaa
def sim(seg, gt):
    label = np.zeros((seg.shape[0],seg.shape[1]))
    for i in range(seg.shape[0]):
        for j in range(seg.shape[0]):
            if seg[i,j,0]==gt[i,j,0] and seg[i,j,1]==gt[i,j,1] and seg[i,j,2]==gt[i,j,2]:
                label[i,j] = 1
    return label

def siam_transform(img,gt,pos,neg,other):
    aug1 = iaa.Sequential(
        iaa.Fliplr(1)
    )
    aug2 = iaa.Sequential(
        iaa.Flipud(1)
    )
    aug3 = iaa.Sequential([
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0)]
    )


    aug = random.randint(0,4)
    if aug==0:
        return img,gt,pos,neg,other
    elif aug==1:
        img = aug1.augment_image(img)
        gt = aug1.augment_image(gt)
        pos = aug1.augment_image(pos)
        neg = aug1.augment_image(neg)
        other = aug1.augment_image(other)
        return img,gt,pos,neg,other
    elif aug==2:
        img = aug2.augment_image(img)
        pos = aug2.augment_image(pos)
        neg = aug3.augment_image(neg)
        gt = aug2.augment_image(gt)
        other = aug2.augment_image(other)
        return img,pos,neg,gt,other
    else:
        img = aug3.augment_image(img)
        pos = aug3.augment_image(pos)
        neg = aug3.augment_image(neg)
        other = aug3.augment_image(other)
        gt = aug3.augment_image(gt)
        return img,gt,pos,neg,other

def transform(img,prelabel,gt):
    aug1 = iaa.Sequential(
        iaa.Fliplr(1)
    )
    aug2 = iaa.Sequential(
        iaa.Flipud(1)
    )

    flr = random.randint(0,1)
    fud = random.randint(0,1)
    fcut = random.randint(0,1)
    if flr==1:
        img = aug1.augment_image(img)
        prelabel = aug1.augment_image(prelabel)
        gt = aug1.augment_image(gt)
        return img,prelabel,gt
    elif fud==1:
        img = aug2.augment_image(img)
        prelabel = aug2.augment_image(prelabel)
        gt = aug2.augment_image(gt)
        return img,prelabel,gt
    elif fcut==1:
        prelabel = cut_postive(prelabel)
        return img,prelabel,gt
    return  img,prelabel,gt
def cut_postive(img):
    cut_ratio = random.uniform(0.1,0.4)
    cut_direction = random.randint(0,3)
    up_x, bot_x,left_y, right_y = 255, 0, 255, 0
    neg = np.zeros((img.shape[0],img.shape[1],3))
    for h in range(neg.shape[0]):
        for w in range(neg.shape[1]):
            for c in range(3):
                neg[h][w][c] = img[h][w][c]
    for h in range(neg.shape[0]):
        for w in range(neg.shape[1]):
            if neg[h][w][0]!=0 or neg[h][w][1]!=0 or neg[h][w][2]!=0:
                if up_x > h:
                    up_x = h
                if bot_x < h:
                    bot_x = h
                if left_y > w:
                    left_y = w
                if right_y < w:
                    right_y = w
    if cut_direction == 0:
        neg[up_x:int((bot_x-up_x)*cut_ratio+up_x)][:][:] = 0
    elif cut_direction == 1:
        neg[int((bot_x-up_x)*cut_ratio+up_x):bot_x][:][:] = 0
    elif cut_direction == 2:
        neg[:][left_y:int((right_y-left_y)*cut_ratio+left_y)][:] = 0
    elif cut_direction == 3:
        neg[:][int((right_y-left_y)*cut_ratio+left_y):right_y][:] = 0
    return neg

if __name__ == "__main__":
    zero = np.zeros((320,320,3))
    anchor = cv2.imread("/Users/lixiaofan/Desktop/分割评估621/data/train/anchor/2008_004730.png")
    seg = cv2.imread("/Users/lixiaofan/Desktop/分割评估621/data/train/negative/2008_004730.png")
    anchor = cv2.resize(anchor,(320,320))
    seg = cv2.resize(seg,(320,320))
    seg[:160,:320,:] = zero[:160,:320,:]
    label = sim(seg,anchor)
    cv2.imwrite("cut.png",seg*255)
    cv2.imwrite("单张相似性标签.png",label*255)

