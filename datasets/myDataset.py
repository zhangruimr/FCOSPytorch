from __future__ import division
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import os
import math
import random
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms as transforms

"""
def single_channel_process(img):
    if len(img.shape) < 3:
        img = np.tile(np.expand_dims(img, -1), (1, 1, 3))
    return img

def bgr2rgb(img):
    img = single_channel_process(img)
    return img[:,:,::-1]
"""
def horizentalFlip(img, boxes):
        if random.random() > 0.5:
            img = img[:, ::-1, :]
            boxes[:, 1] = 1 - boxes[:, 1]
        return img, boxes

def verticalFlip(img, boxes):
    if random.random() > 0.5:
        img = img[::-1, :, :]
        boxes[:, 2] = 1 - boxes[:, 2]
    return img, boxes

def randomRotation(img, boxes):
    _h, _w, _ = img.shape
    angle = int(random.uniform(-10, 10))
    #获得旋转变换矩阵
    rot_matrix = cv2.getRotationMatrix2D((_w // 2, _h // 2), angle, 1)
    #旋转图片
    img = cv2.warpAffine(img, rot_matrix, (_w, _h))

    #计算旋转后的ground-truth
    x, y, w, h = boxes[:, 1]*_w, boxes[:, 2]*_h, boxes[:, 3]*_w, boxes[:, 4]*_h
    xl, yl = x - 0.5 * w, y - 0.5 * h
    xr, yr = x + 0.5 * w, y + 0.5 * h
    spot11 = (rot_matrix[0, 0] * xl + rot_matrix[0, 1] * yl + rot_matrix[0, 2],
              rot_matrix[1, 0] * xl + rot_matrix[1, 1] * yl + rot_matrix[1, 2])
    spot12 = (rot_matrix[0, 0] * xr + rot_matrix[0, 1] * yl + rot_matrix[0, 2],
              rot_matrix[1, 0] * xr + rot_matrix[1, 1] * yl + rot_matrix[1, 2])
    spot21 = (rot_matrix[0, 0] * xl + rot_matrix[0, 1] * yr + rot_matrix[0, 2],
              rot_matrix[1, 0] * xl + rot_matrix[1, 1] * yr + rot_matrix[1, 2])
    spot22 = (rot_matrix[0, 0] * xr + rot_matrix[0, 1] * yr + rot_matrix[0, 2],
              rot_matrix[1, 0] * xr + rot_matrix[1, 1] * yr + rot_matrix[1, 2])

    xmin = np.min(np.stack((spot11[0], spot12[0], spot21[0], spot22[0]), -1), -1)
    ymin = np.min(np.stack((spot11[1], spot12[1], spot21[1], spot22[1]), -1), -1)
    xmax = np.max(np.stack((spot11[0], spot12[0], spot21[0], spot22[0]), -1), -1)
    ymax = np.max(np.stack((spot11[1], spot12[1], spot21[1], spot22[1]), -1), -1)

    center_x = 0.5 * (xmin + xmax) / _w
    center_y = 0.5 * (ymin + ymax) / _h
    center_w = (xmax - xmin) / _w
    center_h = (ymax - ymin) / _h
    boxes = np.stack((boxes[:, 0], center_x, center_y, center_w, center_h), 1)
    return img, boxes

class augment():
    def __init__(self, colorJitter, horizentalFlip, verticalFlip, randomRotation):
        self.colorJitter = colorJitter
        self.horizentalFlip = horizentalFlip
        self.verticalFlip = verticalFlip
        self.randomRotation = randomRotation
    def __call__(self, img, boxes):
        if self.colorJitter:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)(img)

        img = np.array(img)[..., ::-1] / 255
        """
        print(boxes)
        pic = img
        h, w, _ = img.shape
        print(img.shape)
        x, y, _w, _h = boxes[:, 1]*w, boxes[:, 2]*h, boxes[:, 3]*w, boxes[:, 4]*h
        boxes[:, 1] = x - 0.5*_w
        boxes[:, 2] = y - 0.5*_h
        boxes[:, 3] = x + 0.5*_w
        boxes[:, 4] = y + 0.5*_h
        print(boxes)
        for box in boxes:
            #if box[0] != -1:
            x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            #print(x1, y1, x2, y2)
            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.imshow("win", pic)
        cv2.waitKey(0)
"""
        if self.horizentalFlip:
            img, boxes = horizentalFlip(img, boxes)
        if self.verticalFlip:
            img, boxes = verticalFlip(img, boxes)
        if self.randomRotation:
            img, boxes = randomRotation(img, boxes)
        return img, boxes

class TrainDataset(Dataset):
    def __init__(self, trainDirs, size=(800, 1333),  colorJitter=True, horizentalFlip=True, verticalFlip=True, randomRotation=True):
        self.trainDirs = trainDirs
        self.labelsDirs = trainDirs.replace("images", "labels")
        filenames = os.listdir(self.trainDirs)
        self.imgsPath = [os.path.join(self.trainDirs, pic) for pic in filenames]
        self.labelsPath = [os.path.join(self.labelsDirs, label.replace("jpg", "txt").replace("png", "txt").replace("jpeg", "txt")) \
                           for label in filenames]
        self.size = size

        self.augment = augment(colorJitter, horizentalFlip, verticalFlip, randomRotation)
    def __getitem__(self, item):
        imgroad = self.imgsPath[item]
        img = Image.open(imgroad)

        labelroad = self.labelsPath[item]
        label = np.loadtxt(labelroad).astype("float32")
# label是按照 yolo3 标注的格式(类别， center_x, center_y, w, h), 坐标是归一化后的即:
#                                              center_x = x / 图片宽， 其他相同
        label = label.reshape((-1, 5))
        img, label = self.augment(img, label)

        img = t.from_numpy(np.transpose(img, (2, 0, 1))).float()
        label = t.from_numpy(label).float()

        img, label = resize_pad(img, label, self.size)
        _, h, w = img.shape
        x, y, _w, _h = label[:, 1]*w, label[:, 2]*h, label[:, 3]*w, label[:, 4]*h
        label[:, 1] = x - 0.5*_w
        label[:, 2] = y - 0.5*_h
        label[:, 3] = x + 0.5*_w
        label[:, 4] = y + 0.5*_h
        return img, label, imgroad
    def collate_fn(self, batch):
        imgs, labels, roads = list(zip(*batch))
        imgs = t.stack(imgs, 0)
        maxNum = 0
        for label in labels:
            if label.shape[0] > maxNum:
                maxNum = label.shape[0]
        batch_labels = []
        for label in labels:
            shape = label.shape
            boxes = t.ones((maxNum, 5)) * -1
            boxes[0:shape[0], :] = label
            batch_labels.append(boxes)
        labels = t.stack(batch_labels, 0)
        return imgs, labels, roads


    def __len__(self):
        return len(self.imgsPath)

def resize_pad(img, label = None, size=(800, 1333)):
      c, h, w = img.shape
      stride = min(size[0] / h, size[1] / w)
      img = F.interpolate(img.unsqueeze(0), (round(h*stride), round(w*stride))).squeeze(0)
      _, _h, _w = img.shape

      if _h == size[0]:
          dif = (size[1] - _w) // 2
          padding = (dif, size[1] - _w - dif, 0, 0)
          img = F.pad(img.unsqueeze(0), padding).squeeze(0)
          if label is not None:
             label[:, 1] = (label[:, 1]*_w + dif) / size[1]
             label[:, 3] = (label[:, 3]*_w) / size[1]
      elif _w == size[1]:
          dif = (size[0] - _h) // 2
          padding = (0, 0, dif, size[0] - _h - dif)
          img = F.pad(img.unsqueeze(0), padding).squeeze(0)
          if label is not None:
             label[:, 2] = (label[:, 2]*_h + dif) / size[0]
             label[:, 4] = (label[:, 4]*_h) / size[0]
      else:
          print("数据集异常")
          exit()
      return img, label


#测试
if __name__ == "__main__":
    datasets = TrainDataset("D:\VOC2012\images")
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn, drop_last=True)
    for pic, label, road in dataloader:
        #print(pic.shape, label.shape,road)
        pic = pic[0].permute((1,2,0)).contiguous().numpy()
        label = label[0]
        #print(label)

        for box in label:
            if box[0] != -1:
               x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
               print(x1,y1,x2,y2)
               cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 4)

        cv2.imshow("win", pic)
        cv2.waitKey(33)
