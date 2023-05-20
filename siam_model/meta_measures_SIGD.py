import torch
from  torch.utils.data import  DataLoader
import numpy as np
import cv2
from  siam_model.Net_siam  import SiamResNet
import time
from siam_model.data_SIGD import Data
from torch.autograd import Variable
import os
@torch.no_grad()
def evaluate(add,epoch):
    net = SiamResNet()
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    # net.cuda()
    path_model = "/Users/lixiaofan/Desktop/分割评估621/check/"+str(epoch)+".pth"
    net.load_state_dict(torch.load(path_model,map_location=torch.device('cpu')))
    net.eval()
    dataset = Data(add)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers = 0
    )
    total_a = 0
    total_s = 0
    total_t = 0
    total_m = 0
    total_n = 0
    np.set_printoptions(threshold=10000)
    for batch_i, (pos_img ,neg_img,pos_label,neg_label) in enumerate(dataloader):
        pos_f = 0
        pos_b = 0
        t_pos_f = 0.00000001
        t_pos_b = 0.00000001
        neg_f = 0
        neg_b = 0
        t_neg_f = 0.00000001
        t_neg_b = 0.00000001

        pf_score = 0
        pb_score = 0
        nf_score = 0
        nb_score = 0

        pos_img = Variable(pos_img.float())#.cuda()
        neg_img = Variable(neg_img.float())#.cuda()
        pfea_map,nfea_map,p_fea,n_fea = net((pos_img,neg_img))

        zero = torch.zeros(pfea_map.shape)
        one = torch.ones(pfea_map.shape)
        psss = torch.where(pfea_map>=0.5,one,zero)
        psss =psss.sum()
        nsss = torch.where(nfea_map>=0.5,one,zero)
        nsss = nsss.sum()
        if psss>nsss:
            total_n+=1
        for i in range(pfea_map.shape[0]):
            for j in range(pfea_map.shape[1]):
                if pos_img[0,3,i,j]!=0 or pos_img[0,4,i,j]!=0 or pos_img[0,5,i,j]!=0:
                    pf_score+=pfea_map[i,j]
                    if pfea_map[i,j]>=0.5:
                        pos_f+=1
                    t_pos_f+=1
                else:
                    pb_score += pfea_map[i,j]
                    if pfea_map[i,j]>=0.5:
                        pos_b+=1
                    t_pos_b+=1

        for i in range(nfea_map.shape[0]):
            for j in range(nfea_map.shape[1]):
                if neg_img[0,3,i,j]!=0 or neg_img[0,4,i,j]!=0 or neg_img[0,5,i,j]!=0:
                    nf_score+=nfea_map[i,j]
                    if nfea_map[i,j]>=0.5:
                        neg_f+=1
                    t_neg_f+=1
                else:
                    nb_score+=nfea_map[i,j]
                    if nfea_map[i,j]>=0.5:
                        neg_b+=1
                    t_neg_b+=1
        if p_fea>n_fea:
            total_m+=1

        p_score1 = (pf_score/t_pos_f)*0.5+(pb_score/t_pos_b)*0.5
        n_score1 = (nf_score/t_neg_f)*0.5+(nb_score/t_neg_b)*0.5
        pos_score = (pos_f/t_pos_f)*0.5 + (pos_b/t_pos_b)*0.5
        neg_score = (neg_f / t_neg_f) * 0.5 + (neg_b / t_neg_b) * 0.5
        if pos_score>neg_score:
            total_a+=1
        if p_score1>n_score1:
            total_t+=1
        total_s+=1
        print(" poss:",pos_score," negs:",neg_score," pos1:",p_score1," neg1:",n_score1," pos2:",p_fea," neg2:",n_fea)
    print("avgacc:",total_n/total_s,total_a/total_s, total_t/total_s,total_m/total_s)
if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/分割评估621/data/test/","100_16")