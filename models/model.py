import torch as t
import torch.nn as nn
from subModels import *
from loss import loss
#初始化模型参数

#FCOS基本模块，训练测试分别进一步包装
class FCOS(nn.Module):
    def __init__(self, backbone = 101, classNum = 90, weights=None):
        super(FCOS, self).__init__()
        self.classNum = classNum

        self.backbone = Backbone(backbone, weights)
        #resnet101是[512, 1024, 2048]
        self.fpn = FPN([512, 1024, 2048])
        self.detectHead = detectHead(classNum=self.classNum)


    def forward(self, inputs):
        out = self.backbone(inputs)
        out = self.fpn(out)
        out = self.detectHead(out)
        return out
class genTarget():
    def __init__(self, strides, limitRanges):
        self.strides = strides
        self.limtRanges = limitRanges
    def forward(self, inputs, boxes):
        cls, cnt, reg = inputs
        fpn_cls_target = []
        fpn_cnt_target = []
        fpn_reg_target = []
        for i in range(len(cls)):
            layer_out = [cls[i], cnt[i], reg[i]]
            layer_target = generate_layer_target(layer_out, boxes, self.strides[i], self.limtRanges[i])
            fpn_cls_target.append(layer_target[0])
            fpn_cnt_target.append(layer_target[1])
            fpn_reg_target.append(layer_target[2])

        return t.cat(fpn_cls_target, 1), t.cat(fpn_cnt_target, 1), t.cat(fpn_reg_target, 1)
def boxes_sorted(box):
    area = (box[:, 3] - box[:, 1]) * (box[:, 4] - box[:, 2])
    index = t.argsort(area, descending=True)
    return box[index, :]

def generate_layer_target(layer_out, boxes, stride, limitRange):
    cls, cnt, reg = layer_out
    batch = cls.shape[0]
    classNum = cls.shape[-1]
    cls = cls.permute((0, 2, 3, 1)).contiguous().reshape((batch, -1, classNum))
    cnt = cnt.permute((0, 2, 3, 1)).contiguous().reshape((batch, -1, 1))

    grid = generate_grid(reg, stride)
    reg = reg.permute((0, 2, 3, 1)).contiguous().reshape((batch, -1, 4))

    all_cls = []
    all_cnt = []
    all_reg = []
    for i in range(cls.shape[0]):
        single_cls = cls[i]
        single_cnt = cnt[i]
        single_reg = reg[i]
        box = boxes[i]
        box = boxes_sorted(box[box[:, 0] != -1])
        n, _ = box.shape
        x = grid[:, 0]
        y = grid[:, 1]
        h_w = x.shape[0]
        l_off = x[:, None].repeat((1, n)) - box[:, 1][None, :].repeat((h_w, 1))#(h*w, n) - (h*w, n) -> (h*w, n)
        t_off = y[:, None].repeat((1, n)) - box[:, 2][None, :].repeat((h_w, 1))
        r_off = box[:, 3][None, :].repeat((h_w, 1)) - x[:, None].repeat((1, n))
        b_off = box[:, 4][None, :].repeat((h_w, 1)) - y[:, None].repeat((1, n))

        off = t.stack((l_off, t_off, r_off, b_off), -1) #(h*w, n, 4)
        off_min = t.min(off, -1)[0]#(h*w, n)
        off_max = t.max(off, -1)[0]
        mask_in_box = off_min > 0
        mask_in_level = off_max > limitRange[0] & off_max <= limitRange[0]
        mask = mask_in_box & mask_in_level
        val, index = t.max(mask, -1)
        pos = val > 0
        cls_assign = box[index, 0].long()
        reg_assign = box[index, 1:].float()

        l_off = x - reg_assign[:, 1]
        t_off = y - reg_assign[:, 2]
        r_off = reg_assign[:, 3] - x
        b_off = reg_assign[:, 4] - y
        reg_target = t.stack((l_off, t_off, r_off, b_off))

        centerness = t.sqrt((t.min(l_off, r_off) / t.clamp(t.max(l_off, r_off), min=1e-8)) * (t.min(t_off, b_off) / t.clamp(t.max(t_off, b_off), min=1e-8)))

        cnt_target = centerness.reshape(-1,1)

        cls_target = t.zeros(single_cls.shape).float()
        if t.cuda.is_available():
            cls_target = cls_target.cuda()
        seq = t.arange(0, cls_target.shape[0]).long()
        if t.cuda.is_available():
            seq = seq.cuda()
        cls_target[seq[pos], cls_assign[pos]] = 1
        all_cls.append(cls_target)
        all_cnt.append(cnt_target)
        all_reg.append(reg_target)

    return t.stack(all_cls, 0), t.stack(all_cnt, 0), t.stack(all_reg, 0)


"""
    _, n, c = boxes.shape
    x = grid[:, 0]
    y = grid[:, 1]

    l_off = x[None, :, None].repeat((batch, 1, n)) - boxes[..., 0][:, None, :]
    t_off = y[None, :, None].repeat((batch, 1, n)) - boxes[..., 1][:, None, :]
    r_off = boxes[..., 2][:, None, :] - x[None, :, None].repeat((batch, 1, n))
    b_off = boxes[..., 3][:, None, :] - y[None, :, None].repeat((batch, 1, n))

    off = t.stack((l_off, t_off, r_off, b_off), -1)
    off_min = t.min(off, -1)[0]
    off_max = t.max(off, -1)[0]
    mask_in_box = off_min > 0
    mask_in_level = off_max > limitRange[0] & off_max <= limitRange[0]

    area = (off[..., 0] + off[..., 2]) * (off[..., 1] + off[..., 3])
    mask = mask_in_box & mask_in_level
    pos = t.sum(mask.byte(), -1) > 0
    neg = t.sum(mask.byte(), -1) == 0

    cls_select = t.cat((cls[pos], cls[neg]), 0)
    cnt_select = t.cat((cnt[pos], cnt[neg]), 0)
    reg_select = t.cat((reg[pos], reg[neg]), 0)
"""


def generate_grid(feature, stride):
    b, c, h, w = feature.shape
    x = t.arange(0, w*stride, stride).float()
    y = t.arange(0, h*stride, stride).float()
    x, y = t.meshgrid(x, y)
    x = x + stride // 2
    y = y + stride // 2
    grid = t.stack((x,y), -1).reshape((-1, 2))
    if t.cuda.is_available():
        grid = grid.cuda()
    return grid

class FcosTrainer(nn.Module):
    def __init__(self, backbone = 101, classNum = 90):
        super(FcosTrainer, self).__init__()
        self.fcos = FCOS(backbone, classNum)

    def forward(self, inputs, boxes):
         cls, cnt, reg = self.fcos(inputs)







if __name__ == "__main__":
    pass












