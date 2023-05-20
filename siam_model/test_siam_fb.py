import torch
from  torch.utils.data import  DataLoader
import numpy as np
import cv2
from  siam_model.Net_siam  import SiamResNet
import time
from siam_model.data_siam import Data
from torch.autograd import Variable
import os
@torch.no_grad()
def evaluate(add,epoch):
    net = SiamResNet()
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    # net.cuda()
    net.eval()
    path_model = "/Users/lixiaofan/Desktop/分割评估621/check/"+str(epoch)+".pth"
    net.load_state_dict(torch.load(path_model,map_location=torch.device('cpu')))
    dataset = Data(add)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers = 0
    )
    number = 0
    acc_number = 0
    np.set_printoptions(threshold=10000000000)
    for batch_i, (pos_img ,neg_img,pos_label,neg_label) in enumerate(dataloader):
        pos_img = Variable(pos_img.float())#.cuda()
        neg_img = Variable(neg_img.float())#.cuda()
        neg_label = neg_label.float().squeeze()
        pfea_map,nfea_map,p_fea,n_fea = net((pos_img,neg_img))
        total_f = 0.01
        total_b = 0.01
        f= 0
        b = 0
        total_a = 0
        total_s = 0
        for i in range(nfea_map.shape[0]):
            for j in range(nfea_map.shape[1]):
                if neg_img[0,3,i,j]!=0 or neg_img[0,4,i,j]!=0 or neg_img[0,5,i,j]!=0:
                    if nfea_map[i,j]>=0.5:

                        f+=1
                    total_f+=1
                else:
                    if nfea_map[i,j]>=0.5:
                        b+=1
                    total_b+=1
        pl = torch.where(nfea_map >= 0.5)
        prel_pos = len(nfea_map[pl])

        posl = neg_label[pl]
        pl1 = torch.where(posl == 1)
        posl1 = len(posl[pl1])
        nl = torch.where(nfea_map < 0.5)
        prel_neg = len(nfea_map[nl])
        negl = neg_label[nl]
        nl1 = torch.where(negl == 0)
        negl1 = len(negl[nl1])
        total_a += (posl1 + negl1)
        total_s += (prel_pos + prel_neg)
        if ((f/total_f)*0.7+(b/total_b)*0.3) < 0.5:
            acc_number+=1
        number+=1
        print("score0:",((f/total_f)*0.7+(b/total_b)*0.3)," f:",f/total_f," b:",b/total_b," score2:",pfea_map.mean(),nfea_map.mean())
        print("score1:",neg_label.sum()/(320*320),"  acc:",total_a/total_s," pos:",prel_pos/(320*320))
    print(acc_number/number)
if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/分割评估621/data/test/",550)