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
    np.set_printoptions(threshold=10000)
    total_a = 0
    total_s = 0
    total_t = 0
    for batch_i, (pos_img ,neg_img,pos_label,neg_label) in enumerate(dataloader):
        pos_img = Variable(pos_img.float())#.cuda()
        neg_img = Variable(neg_img.float())#.cuda()
        pfea_map,nfea_map,p_fea,n_fea = net((pos_img,neg_img))

        pl = torch.where(pfea_map>=0.5)
        prel_pos = len(pfea_map[pl])
        pos_score = prel_pos/(320*320)
        pl = torch.where(nfea_map >= 0.5)
        prel_pos = len(nfea_map[pl])
        neg_score = prel_pos/(320*320)
        if pos_score>neg_score:
            total_a+=1
        total_s+=1
        print(" poss:",pos_score," negs:",neg_score)
    print("avgacc:",total_a/total_s, total_t/total_s)
if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/分割评估621/data/test/",550)