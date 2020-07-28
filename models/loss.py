import torch as t
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, preds, targets):
        pos_mask = (targets == 1).float()
        #print("num", pos_mask.sum())
        pos_num = t.clamp(pos_mask.sum(), min=1.0)

        preds = t.clamp(preds, min=1e-8, max=1-1e-8)
        print("cls_max:", preds.max())
        print("cls_min:", preds.min())
        print("cls_num:", t.sum(preds>0.6))
        print("\n")
        pos_weights = t.pow(1 - preds, self.gamma) * self.alpha
        pos_loss = - t.log(preds) * pos_weights

        neg_weights = t.pow(preds, self.gamma) * (1 - self.alpha)
        neg_loss = -t.log(1-preds) * neg_weights

        loss = t.where(pos_mask > 0, pos_loss, neg_loss)
        loss = loss.sum() / pos_num
        return loss

"""
class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
    def forward(self, preds, targets, pos_mask):
        print("cnt_max:", preds.max())
        print("cnt_min:", preds.min())
        print("\n")
        pos_preds  = t.clamp(preds[pos_mask], min=1e-8, max=1-1e-8)
        pos_num = len(pos_preds)
        if pos_num < 1:
            pos_num = 1
        #print("num", pos_num)
        pos_targets = targets[pos_mask]
        loss = - t.log(pos_preds) * pos_targets
        loss = loss.sum() / pos_num
        return loss
"""
#regLoss use IouLoss and CenternessLoss use BCE in paper
"""
class GiouLoss(nn.Module):
    def __init__(self):
        super(GiouLoss, self).__init__()
    def forward(self, preds, targets):
        pass
"""

class SmoothL1(nn.Module):
    def __init__(self):
        super(SmoothL1, self).__init__()
    def forward(self, preds, targets, pos_mask):
        preds = preds[pos_mask]
        targets = targets[pos_mask]
        print("reg_out_max:", preds.max())
        print("reg_out_min:", preds.min())
        print("\n")
        #print("reg_labels", t.sum(targets<=0))
        dif = t.abs(preds - targets)
        loss = t.where(dif <= 1, 0.5 * t.pow(dif, 2.0), dif - 0.5)
        pos_num = len(preds)
        if pos_num < 1:
            pos_num = 1
        #print("num", pos_num)
        loss = loss.sum() / pos_num

        return loss
class loss(nn.Module):
    def __init__(self, giou=False):
        super(loss, self).__init__()
        self.giou = giou
        if giou:
            self.regLoss = GiouLoss()
        else:
            self.regLoss = SmoothL1()
        self.clsLoss = FocalLoss()
        #self.cntLoss = BinaryCrossEntropy()
        self.cntLoss = SmoothL1()
    def forward(self, preds, targets):
        cls, cnt, reg = preds
        cls_gt, cnt_gt, reg_gt = targets
        #print("gt", t.sum((cls_gt.sum(-1) == 1).sum()))
        #print("gt2", t.sum((cls_gt.sum(-1) > 1).sum()))
        pos_mask = cls_gt.sum(-1) > 0
        #cnt_loss shoud use BCE and reg_loss use GIOU
        cls_loss = self.clsLoss(cls, cls_gt)
        cnt_loss = self.cntLoss(cnt, cnt_gt, pos_mask)
        reg_loss = self.regLoss(reg, reg_gt, pos_mask)

        print("cls_loss:", cls_loss)
        print("cnt_loss:", cnt_loss)
        print("reg_loss:", reg_loss)

        loss = cls_loss + cnt_loss + 0.01 * reg_loss
        return loss
