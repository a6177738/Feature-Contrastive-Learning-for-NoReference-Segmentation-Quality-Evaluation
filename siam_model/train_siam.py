import torch
from  siam_model.Net_siam import SiamResNet
from siam_model.Contrastive_loss import Contrastive_loss
import torch.backends.cudnn as cudnn
from torch import nn
import time
from siam_model.data_siam import Data
from torch.autograd import Variable
batch_size = 32
num_worker = 0
lr = 4e-3
momentum = 0.9
weight_decay = 0.0005
epochs  = 300
root= "data/train/"   #train root
from model.test import evaluate

if __name__=='__main__':
    dataset = Data(root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
    )
    model = SiamResNet()
    model = torch.nn.DataParallel(model, device_ids=list(range(0,2)))
    model.cuda()
    cudnn.benchmark =True
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        for batch_i, (pos, neg, pos_label, neg_label) in enumerate(dataloader):
            start_time = time.time()
            batches_done = batch_i + 1
            img_pos = Variable(pos.float()).cuda()
            img_neg = Variable(neg.float()).cuda()
            pos_label = pos_label.float().cuda()
            neg_label = neg_label.float().cuda()
            pos_score,neg_score,pos_feamap, neg_feamap, = model((img_pos, img_neg))
            Pixelloss = nn.BCELoss()
            loss_pos = Pixelloss(pos_score,pos_label)
            loss_neg = Pixelloss(neg_score,neg_label)
            pixelloss = loss_pos + loss_neg
            Contrastiveloss = Contrastive_loss()
            contrastive_loss = Contrastiveloss.compute(pos_feamap,neg_feamap,neg_label)
            loss  = pixelloss+contrastive_loss
            end_time = time.time()
            t = end_time - start_time
            loss.backward()
            if batches_done % 2:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            print("epoch:", epoch, " batch_i:", batch_i, " time:", t, " loss:", loss," px:", pixelloss, " contras:",loss_neg)
        if epoch%10==0:
            parapth = "check/" + str(epoch) + ".pth"
            torch.save(model.state_dict(), parapth)
            evaluate("data/test/",epoch)
