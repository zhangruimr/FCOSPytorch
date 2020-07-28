import torch as t
import torch.nn as nn
from subModels import *
from datasets.myDataset import *
import numpy as np
from utils.functions import *
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

class generateTarget(nn.Module):
    def __init__(self, strides, limitRanges):
        super(generateTarget, self).__init__()
        self.strides = strides
        self.limtRanges = limitRanges
    def forward(self, inputs, boxes):
        cls, cnt, reg = inputs
        fpn_cls_target = []
        fpn_cnt_target = []
        fpn_reg_target = []

        new_cls = []
        new_cnt = []
        new_reg = []
        for i in range(len(cls)):
            layer_out = [cls[i], cnt[i], reg[i]]
            layer_target = generate_layer_target(layer_out, boxes, self.strides[i], self.limtRanges[i])

            fpn_cls_target.append(layer_target[0])
            fpn_cnt_target.append(layer_target[1])
            fpn_reg_target.append(layer_target[2])

            new_cls.append(layer_target[3])
            new_cnt.append(layer_target[4])
            new_reg.append(layer_target[5])
        return t.cat(fpn_cls_target, 1), t.cat(fpn_cnt_target, 1), t.cat(fpn_reg_target, 1), t.cat(new_cls, 1), t.cat(new_cnt, 1), t.cat(new_reg, 1)

class FcosTrainer(nn.Module):
    def __init__(self, backbone = 101, weights=None, classNum = 20, strides=[8,16,32,64,128], limitRanges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]):
        super(FcosTrainer, self).__init__()
        self.fcos = FCOS(backbone, weights=weights, classNum=classNum)
        self.strides = strides
        self.limitRanges = limitRanges
        self.generateTargets = generateTarget(self.strides, self.limitRanges)
    def forward(self, inputs, boxes):
         outputs = self.fcos(inputs)
         cls_label, cnt_label, reg_label, cls, cnt, reg = self.generateTargets(outputs, boxes)

         outputs = [cls, cnt, reg]
         targets = [cls_label, cnt_label, reg_label]
         """
         for i in range(len(targets)):
             print("target", targets[i].shape)
             print("feature", outputs[i].shape)
         s"""
         return outputs, targets

class postProcessing(nn.Module):
    def __init__(self, strides, limitRanges):
        super(postProcessing, self).__init__()
        self.strides = strides
    def forward(self, inputs):
        cls, cnt, reg = inputs
        new_cls = []
        new_cnt = []
        new_reg = []
        for i in range(len(cls)):
            layer_out = [cls[i], cnt[i], reg[i]]
            new_results = layer_post_processing(layer_out, self.strides[i])
            #print(i, new_results[0].shape,new_results[1].shape,new_results[2].shape)
            new_cls.append(new_results[0])
            new_cnt.append(new_results[1])
            new_reg.append(new_results[2])
        cls = t.cat(new_cls, 1)
        cnt = t.cat(new_cnt, 1)
        reg = t.cat(new_reg, 1)

        batch = cls.shape[0]

        all_results = []
        for i in range(batch):
            cls_out = cls[i]
            cnt_out = cnt[i]
            reg_out = reg[i]
            sscls_out = t.sqrt(cls_out * cnt_out)
            score, kind = cls_out.max(-1)

            mask = score > 0.05

            cls_out = cls_out[mask]
            cnt_out = cnt_out[mask]
            reg_out = reg_out[mask]
            score = score[mask]
            kind = kind[mask].float()

            #score = t.sqrt(score * cnt_out)
            #print(reg_out.shape, kind.unsqueeze(-1).shape, score.unsqueeze(-1).shape)
            result = t.cat((reg_out, kind.unsqueeze(-1), score.unsqueeze(-1)), -1)
            objects = nms(result)
            #if objects is not None:
                #objects = clip_box(objects, self.size)
            all_results.append(objects)
        return all_results

class FcosDetector(nn.Module):
    def __init__(self, backbone=101, classNum=20, strides=[8, 16, 32, 64, 128], limitRanges=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]):
        super(FcosDetector, self).__init__()
        self.strides = strides
        self.limitRanges = limitRanges

        self.fcos = FCOS(backbone, classNum=classNum)
        self.post_process = postProcessing(self.strides, self.limitRanges)

    def forward(self, inputs):
        outputs = self.fcos(inputs)
        results = self.post_process(outputs)

        return results
if __name__ == "__main__":
    datasets = TrainDataset("/home/zr/VOC/VOC2012/images")
    dataloader = DataLoader(datasets, batch_size=2, shuffle=True, collate_fn=datasets.collate_fn, drop_last=True)
    #model = FcosTrainer()
    model = FCOS()
    model = model.cuda()

    for imgs, labels ,roads in dataloader:
        if t.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        x = model(imgs)
        #output, labels = model(imgs, labels)














