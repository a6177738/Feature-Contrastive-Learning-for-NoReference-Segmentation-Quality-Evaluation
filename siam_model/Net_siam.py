import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tm
import numpy as np



class SiamResNet(nn.Module):
    def __init__(self):
        super(SiamResNet,self).__init__()
        self.resnet = tm.resnet50()
        self.bn = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.features1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.score_32s = nn.Conv2d(512 * 4,
                                   256,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256*4 ,
                                   256,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 * 4,
                                  256,
                                  kernel_size=1)
        self.tmconv = nn.Conv2d(256,
                                  1,
                                  kernel_size=1)
    def forward(self, x):
        pos, neg = x
        pos = self.features1(pos)
        pos = self.resnet.maxpool(pos)
        pos = self.resnet.layer1(pos)
        pos = self.resnet.layer2(pos)
        pos_8 = self.score_8s(pos)
        pos = self.resnet.layer3(pos)
        pos_16 = self.score_16s(pos)
        pos = self.resnet.layer4(pos)
        pos_32 = self.score_32s(pos)
        pos_16_spatial_dim = pos_16.size()[2:]
        pos_8_spatial_dim = pos_8.size()[2:]
        pos_16 += nn.functional.interpolate(pos_32,
                                                size=pos_16_spatial_dim,
                                                mode="bilinear",
                                                align_corners=True)

        pos_8 += nn.functional.interpolate(pos_16,
                                               size=pos_8_spatial_dim,
                                               mode="bilinear",
                                               align_corners=True)

        pos_upsampled = nn.functional.interpolate(pos_8,
                                                     size=(320,320),
                                                     mode="bilinear",
                                                     align_corners=True)
        pos_sim = self.tmconv(pos_upsampled)
        pos_sim = torch.sigmoid(pos_sim).squeeze()
        pos_score = pos_sim.mean()


        neg = self.features1(neg)
        neg = self.resnet.maxpool(neg)
        neg = self.resnet.layer1(neg)
        neg = self.resnet.layer2(neg)
        neg_8 = self.score_8s(neg)
        neg = self.resnet.layer3(neg)
        neg_16 = self.score_16s(neg)
        neg = self.resnet.layer4(neg)
        neg_32 = self.score_32s(neg)
        neg_16_spatial_dim = neg_16.size()[2:]
        neg_8_spatial_dim = neg_8.size()[2:]
        neg_16 += nn.functional.interpolate(neg_32,
                                            size=neg_16_spatial_dim,
                                            mode="bilinear",
                                            align_corners=True)

        neg_8 += nn.functional.interpolate(neg_16,
                                           size=neg_8_spatial_dim,
                                           mode="bilinear",
                                           align_corners=True)

        neg_upsampled = nn.functional.interpolate(neg_8,
                                                  size=(320, 320),
                                                  mode="bilinear",
                                                  align_corners=True)
        neg_sim = self.tmconv(neg_upsampled)
        neg_sim  = torch.sigmoid(neg_sim).squeeze()
        neg_score = neg_sim.mean()
        return pos_sim,neg_sim,pos_score,neg_score

if __name__ == '__main__':
    model = SiamResNet(8)
    pos = torch.zeros((8,4,512,512))
    neg = torch.zeros((8,4,512,512))
    pos_score,neg_socre = model((pos,neg))
    print(pos_score.shape)