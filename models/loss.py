import torch as t
import numpy as np
import torch.nn as nn
from utils.functions import *
def focalLoss(pos_output, neg_output, pos_label, neg_label, alpha=0.25, gamma=2.0):

    pos_num = len(pos_output)
    output = t.cat((pos_output, neg_output), 0)
    target = t.cat((pos_label, neg_label), 0)
    if pos_num < 1:
        pos_num = 1
    pos_weights = t.pow(1-output, gamma) * alpha * target
    pos_loss = - t.log(output) * pos_weights

    neg_weights = t.pow(output, gamma) * (1 - alpha) * (1 - target)
    neg_loss = - t.log(1.0 - output) * neg_weights
    loss = (pos_loss.sum() + neg_loss.sum()) / pos_num
    return loss
def smoothLoss(output, label):
    dif = t.abs(output - label)
    loss = t.where(dif <= 1, 0.5 * t.pow(dif, 2.0), dif - 0.5)
    pos_num = len(output)
    if pos_num < 1:
        pos_num = 1

    loss = loss.sum() / pos_num
    return loss

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
    def forward(self, cls, reg, labels, anchors, alpha=0.25, gamma=2.0):
        batch = cls.shape[0]
        cls_loss = t.zeros((1), requires_grad=True)
        reg_loss = t.zeros((1), requires_grad=True)

        if t.cuda.is_available():
            cls_loss = cls_loss.cuda(2)
            reg_loss = reg_loss.cuda(2)
        for i in range(batch):
            cls_output = t.clamp(cls[i], min=1e-4, max=1.0 - 1e-4)

            reg_output = reg[i]
            anchor = anchors[i]
            label = labels[labels[:, 0].int()==i]

        #have no object
            if label.shape[0] < 1:
                print("batch-{}:no object!!".format(i))
                reg_loss = reg_loss + 0

                weights = t.pow(cls_output, gamma) * (1 - alpha)
                loss = - t.log(1.0 - cls_output) * weights
                cls_loss = cls_loss + loss.sum()
                continue

            ious = []
            for box in label:
                ious.append(iou(anchor, box[2:]))
            ious = t.stack(ious, 1)
            max_val, max_id = t.max(ious, 1)
            mask_pos = max_val > 0.5
            mask_neg = max_val < 0.4


            if t.sum(mask_pos) == 0:
                print("no pos_sampling anchors!!")
                continue
            if t.sum(mask_neg) == 0:
                print("no neg_sampling anchors!!")
                continue
            pos_cls = cls_output[mask_pos]
            neg_cls = cls_output[mask_neg]
            pos_labels = t.zeros(pos_cls.shape).float()
            neg_labels = t.zeros(neg_cls.shape).float()
            if t.cuda.is_available():
                pos_labels = pos_labels.cuda(2)
                neg_labels = neg_labels.cuda(2)

            seq = t.arange(0, len(pos_labels)).long()
            if t.cuda.is_available():
                seq = seq.cuda(2)

            cls_index = label[:, 1][max_id[mask_pos].long()].long()

            pos_labels[seq, cls_index] = 1


            cls_loss = cls_loss + focalLoss(pos_cls,  neg_cls, pos_labels, neg_labels)


            pos_anchor = anchor[mask_pos]
            pos_gt = label[max_id[mask_pos].long(), :]

            pos_ctrx = (pos_anchor[:, 0] + pos_anchor[:, 2]) * 0.5
            pos_ctry = (pos_anchor[:, 1] + pos_anchor[:, 3]) * 0.5
            pos_w = pos_anchor[:, 2] - pos_anchor[:, 0]
            pos_h = pos_anchor[:, 3] - pos_anchor[:, 1]

            pos_gtx = (pos_gt[:, 2] + pos_gt[:, 4]) * 0.5
            pos_gty = (pos_gt[:, 3] + pos_gt[:, 5]) * 0.5
            pos_gtw = pos_gt[:, 4] - pos_gt[:, 2]
            pos_gth = pos_gt[:, 5] - pos_gt[:, 3]

            dx = (pos_gtx - pos_ctrx) / pos_w
            dy = (pos_gty - pos_ctry) / pos_h
            dw = t.log(pos_gtw / pos_w)
            dh = t.log(pos_gth / pos_h)

            target = t.stack((dx, dy, dw, dh), 1)

            reg_loss = reg_loss + smoothLoss(reg_output[mask_pos], target)

        all_loss = (reg_loss + cls_loss) / batch

        return all_loss









