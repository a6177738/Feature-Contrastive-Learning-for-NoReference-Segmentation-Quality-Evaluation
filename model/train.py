import torch
from model.net import ResFCN
import time
from model.data import Data
from torch.autograd import Variable
batch_size = 16
num_worker = 0
lr = 4e-3
momentum = 0.9
weight_decay = 0.0005
epochs  = 300
root= "data/train/"
from model.test import evaluate

if __name__=='__main__':
    dataset = Data(root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
    )
    Net = ResFCN()
    optimizer = torch.optim.SGD(Net.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    for epoch in range(epochs):
        Net.train()
        for batch_i, (img, label) in enumerate(dataloader):
            start_time = time.time()
            batches_done = batch_i + 1
            img = Variable(img.float())
            label = label.float()
            pre_label = Net(img)
            Loss = torch.nn.BCELoss()
            loss = Loss(pre_label,label)
            end_time = time.time()
            t = end_time - start_time
            loss.backward()
            if batches_done % 2:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            print("epoch:", epoch, " batch_i:", batch_i, " time:", t, " loss:", loss)
        parapth = "check/" + str(epoch) + ".pth"
        torch.save(Net.state_dict(), parapth)
        evaluate("data/test/",epoch)
