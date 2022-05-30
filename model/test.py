import torch
from  torch.utils.data import  DataLoader
import numpy as np
from  model.net import ResFCN
from model.data import Data
from torch.autograd import Variable
@torch.no_grad()
def evaluate(add,epoch):
    Net = ResFCN()
    #net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    #net.cuda()
    Net.eval()
    path_model = "/Users/lixiaofan/Desktop/分割评估621/check/"+str(epoch)+".pth"
    Net.load_state_dict(torch.load(path_model,map_location='cpu'))
    dataset = Data(add)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers = 0
    )
    np.set_printoptions(threshold=10000)
    a_pos = 0
    s_pos = 0
    for batch_i, (img ,label) in enumerate(dataloader):
        img = Variable(img.float())#.cuda()
        label = label.float()
        fea_map = Net(img)
        pl = torch.where(fea_map>=0.5)
        prel_pos = len(fea_map[pl])
        posl = label[pl]
        pl1 = torch.where(posl==1)
        posl1 = len(posl[pl1])
        nl = torch.where(fea_map < 0.5)
        prel_neg = len(fea_map[nl])
        negl = label[nl]
        nl1 = torch.where(negl == 0)
        negl1 = len(negl[nl1])
        a_pos += (posl1+negl1)
        s_pos += (prel_pos+prel_neg)

    print("avgacc:",a_pos/s_pos)
if __name__ == "__main__":
    evaluate("data/test","52")