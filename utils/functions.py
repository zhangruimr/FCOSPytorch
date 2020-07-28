import torch as t
import torch.nn as nn
import numpy as np
import math
def boxes_sorted(box):
    area = (box[:, 3] - box[:, 1]) * (box[:, 4] - box[:, 2])
    index = t.argsort(area, descending=True)
    return box[index, :]

def generate_grid(feature, stride):
    b, c, h, w = feature.shape
    x = t.arange(0, w*stride, stride).float()
    y = t.arange(0, h*stride, stride).float()
    x, y = t.meshgrid(x, y)
    x = x.t() + stride // 2
    y = y.t() + stride // 2
    grid = t.stack((x, y), -1).reshape((-1, 2))
    if t.cuda.is_available():
        grid = grid.cuda()
    return grid

def generate_layer_target(layer_out, boxes, stride, limitRange):
    cls, cnt, reg = layer_out
    batch = cls.shape[0]
    classNum = cls.shape[1]
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
        #按照box面积大小降序排序，后面为特征点筛选模糊时选择较小box
        box = boxes_sorted(box[box[:, 0] >= 0])
        #print( box[:,0])
        #print("i:", i)
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

        mask_in_level = (off_max > limitRange[0]) & (off_max <= limitRange[1])
        mask = mask_in_box & mask_in_level

        val, index = t.max(mask, -1)
        pos = val > 0
        cls_assign = box[index, 0].long()
        reg_assign = box[index, 1:].float()

        l_off = x - reg_assign[:, 0]
        t_off = y - reg_assign[:, 1]
        r_off = reg_assign[:, 2] - x
        b_off = reg_assign[:, 3] - y
        reg_target = t.stack((l_off, t_off, r_off, b_off), -1)

        centerness = t.sqrt((t.clamp(t.min(l_off, r_off), min=1e-8) / t.clamp(t.max(l_off, r_off), min=1e-8)) * (t.clamp(t.min(t_off, b_off), min=1e-8) / t.clamp(t.max(t_off, b_off), min=1e-8)))

        cnt_target = centerness.reshape(-1, 1)

        cls_target = t.zeros(single_cls.shape).float()
        if t.cuda.is_available():
            cls_target = cls_target.cuda()
        seq = t.arange(0, cls_target.shape[0]).long()
        if t.cuda.is_available():
            seq = seq.cuda()

        cls_target[seq[pos], cls_assign[pos]] = 1

        #print( cls_target.max(-1)[1], cls_target.max(-1)[0], box)
        #print("i:", i)
        #print("cls333", cls_target.max(-1)[0])
        #print("cls", cls_target.sum(-1))
        #print("cls", t.sum(cls_target))
        #print("reg", reg_target)
        #print("cnt", cnt_target)
        all_cls.append(cls_target)
        all_cnt.append(cnt_target)
        all_reg.append(reg_target)

    return t.stack(all_cls, 0), t.stack(all_cnt, 0), t.stack(all_reg, 0), cls, cnt, reg

def layer_post_processing(output, stride):
    cls, cnt, reg = output
    b, c, h, w = cls.shape
    grid = generate_grid(cls, stride)
    cls = cls.permute((0, 2, 3, 1)).contiguous().reshape((b, -1, c))
    cnt = cnt.permute((0, 2, 3, 1)).contiguous().reshape((b, -1, 1))
    reg = reg.permute((0, 2, 3, 1)).contiguous().reshape((b, -1, 4))


    x = grid[:, 0]
    y = grid[:, 1]

    new_reg = []
    for i in range(b):
        reg_out = reg[i]

        reg_out[:, 0] = x - reg_out[:, 0]
        reg_out[:, 1] = y - reg_out[:, 1]
        reg_out[:, 2] = reg_out[:, 2] + x
        reg_out[:, 3] = reg_out[:, 3] + y

        new_reg.append(reg_out)
    new_reg = t.stack(new_reg, 0)

    return cls, cnt, new_reg

def iou(anchors, label):
    a_x1, a_y1, a_x2, a_y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_x1, gt_y1, gt_x2, gt_y2 = label[0], label[1], label[2], label[3]

    insection = t.clamp(t.min(a_x2, gt_x2) - t.max(a_x1, gt_x1), min=0) * t.clamp(t.min(a_y2, gt_y2) - t.max(a_y1, gt_y1), min=0)

    union  = (a_x2 - a_x1) * (a_y2 - a_y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - insection

    iou = insection / union

    return iou
def nms(results, score_thresh=0.5, iou_thresh=0.5):


    filter_mask = results[:, 5] > score_thresh
    if not t.sum(filter_mask) > 0:
        return None
    res = results[filter_mask]

    seq = t.argsort(res[:, -1], descending=True)
    res = res[seq, :].float()
    #print(res)
    objects = []
    while len(res) > 0:
        box = res[0]
        objects.append(box)
        ious = iou(res[:, 0:4], box[0:4])
        iou_mask = ious > iou_thresh
        cls_mask = res[:, -2] == box[-2]

        mask = iou_mask & cls_mask
        mask = ~mask
        res = res[mask]

    objects = t.stack(objects, 0)
    return objects

def clip_box(reg_output, size):
    h, w = size
    bottom = t.zeros(reg_output.shape)
    w_top = t.ones((reg_output.shape[0], )) * w
    h_top = t.ones((reg_output.shape[0], )) * h
    if t.cuda.is_available():
        bottom = bottom.cuda()
        w_top = w_top.cuda()
        h_top = h_top.cuda()
    reg_output[:, 0:5] = t.where(reg_output[:, 0:5] < 0, bottom[:, 0:5], reg_output[:, 0:5])
    reg_output[:, 0] = t.where(reg_output[:, 0] > w, w_top, reg_output[:, 0])
    reg_output[:, 1] = t.where(reg_output[:, 1] > h, h_top, reg_output[:, 1])
    reg_output[:, 2] = t.where(reg_output[:, 2] > w, w_top, reg_output[:, 2])
    reg_output[:, 3] = t.where(reg_output[:, 3] > h, h_top, reg_output[:, 3])
    return reg_output

def cal_box(detection, image_shape, size):
    max_h, max_w = size
    h, w, c = image_shape
    min_stride = min(max_h / h, max_w / w)
    _h, _w = round(h * min_stride), round(w * min_stride)
    #print(h, w)
    #print(_h, _w)
    if _h == max_h:
        dif = (max_w - _w) // 2
        #print("dif",dif)
        detection[:, 0] = detection[:, 0] - dif
        detection[:, 2] = detection[:, 2] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride

    elif _w == max_w:
        dif = (max_h - _h) // 2
        #print("dif", dif)
        detection[:, 1] = detection[:, 1] - dif
        detection[:, 3] = detection[:, 3] - dif
        detection[:, 0:4] = detection[:, 0:4] / min_stride
    return detection