import torch
from  torch.utils.data import  DataLoader
import numpy as np
import cv2
from  siam_model.Net_siam import SiamResNet
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
        batch_size=8,
        shuffle=False,
        num_workers = 0
    )
    np.set_printoptions(threshold=10000)
    total_a = 0
    total_s = 0
    a_pos = 0
    s_pos = 0
    a_neg = 0
    s_neg = 0

    for batch_i, (pos_img ,neg_img,pos_label,neg_label) in enumerate(dataloader):
        pos_img = Variable(pos_img.float(),requires_grad=False)
        neg_img = Variable(neg_img.float(),requires_grad=False)
        pfea_map,nfea_map,p_fea,n_fea = net((pos_img,neg_img))
        pl = torch.where(pfea_map>=0.5)
        prel_pos = len(pfea_map[pl])
        posl = pos_label[pl]
        pl1 = torch.where(posl==1)
        posl1 = len(posl[pl1])
        nl = torch.where(pfea_map < 0.5)
        prel_neg = len(pfea_map[nl])
        negl = pos_label[nl]
        nl1 = torch.where(negl == 0)
        negl1 = len(negl[nl1])
        a_pos += (posl1+negl1)
        s_pos += (prel_pos+prel_neg)
        total_a += (posl1+negl1)
        total_s += (prel_pos+prel_neg)

        pl = torch.where(nfea_map>=0.5)
        prel_pos = len(nfea_map[pl])
        posl = neg_label[pl]
        pl1 = torch.where(posl==1)
        posl1 = len(posl[pl1])
        nl = torch.where(nfea_map < 0.5)
        prel_neg = len(nfea_map[nl])
        negl = neg_label[nl]
        nl1 = torch.where(negl == 0)
        negl1 = len(negl[nl1])
        a_neg += (posl1+negl1)
        s_neg += (prel_pos+prel_neg)
        total_a += (posl1+negl1)
        total_s += (prel_pos+prel_neg)
        print("score:",a_pos/s_pos, "  ",a_neg/s_neg)

    print("avgacc:",total_a/total_s, " poss:",a_pos/s_pos," negs:",a_neg/s_neg)
if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/分割评估621/data/test/","35")