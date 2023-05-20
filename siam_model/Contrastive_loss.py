import torch
class Contrastive_loss():
    def __init__(self):
        super(Contrastive_loss, self).__init__()

    def compute(self, pos_feamap, neg_feamap, neg_label):

        feamap = pos_feamap*neg_feamap #[16,320,320,256]
        feamap = torch.sum(feamap,dim=1)
        pos_square = pos_feamap * pos_feamap
        neg_square = neg_feamap * neg_feamap
        pos_sqrt = torch.sqrt(torch.sum(pos_square,dim=1))
        neg_sqrt = torch.sqrt(torch.sum(neg_square, dim=1))
        sim = feamap/(pos_sqrt*neg_sqrt)
        score = neg_label-sim
        contras_loss = torch.abs(score)
        sum_contras_loss = contras_loss.sum()


        return sum_contras_loss
if __name__ =="__main__":
     a = torch.zeros((16,256,320,320))
     b = torch.zeros((16,256,320,320))
     c = torch.zeros((16,320,320))
     contras = Contrastive_loss()
     l = contras.compute(a,b,c)