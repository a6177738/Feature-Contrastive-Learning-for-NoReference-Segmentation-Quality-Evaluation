import torch
from  torch.utils.data import  DataLoader
import numpy as np
from  model.net import ResFCN
from model.data import Data
from torch.autograd import Variable
import cv2
@torch.no_grad()
def evaluate(add,epoch):
    Net = ResFCN()
    #net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    #net.cuda()
    Net.eval()
    path_model = "/Users/lixiaofan/Desktop/分割评估621/check/"+str(epoch)+".pth"
    Net.load_state_dict(torch.load(path_model,map_location='cpu'))
    sourcefile = "/Users/lixiaofan/Desktop/分割评估621/data/train/"
    original = cv2.imread(sourcefile+"original/2008_004730.jpg")
    original = cv2.resize(original,(320,320)).transpose(2,0,1)
    #2009_004994, 000723
    segmentation = cv2.imread(sourcefile+"negative/2008_004730.png")
    segmentation = cv2.resize(segmentation,(320,320)).transpose(2,0,1)
    zero = np.zeros((3,320,320))
    segmentation[:,:160,:320] = zero[:,:160,:320]
    # cam = cv2.imread(sourcefile+"cam/2009_003707.png",0)
    # cam = cv2.resize(cam,(320,320))

    img = torch.from_numpy(np.concatenate((original,segmentation),0)).unsqueeze(dim=0)
    img = Variable(img.float())#.cuda()
    fea_map = Net(img).numpy()
    # w = ((fea_map*cam).sum())/(cam.sum())
    # if w>0.5:
    print(fea_map.mean())
    cv2.imwrite('单张相似性.png',fea_map*255)
    # else:
    #    print(fea_map.mean()*w)

if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/分割评估621/data/test_new","1_001posneg")