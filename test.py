import torch as t
import numpy as np
import PIL.Image as Image
from torchvision import transforms
import random
import numpy as np
import cv2
import os
import shutil
x = t.Tensor([3, 4, 4, 4, 3, 5, 3, 5])
c = t.argsort(x, descending=True).long()
x = x.repeat((2,1)).t()
f = t.Tensor([True, False])
print(f.byte())
print(x[c, :])
"""
pic = cv2.imread("test.jpg")


cv2.rectangle(pic,(100, 100), (200, 300), (255, 0, 0), 10)
rot_matrix = cv2.getRotationMatrix2D((pic.shape[0]//2, pic.shape[1]//2), -58, 1)

spot11 = (int(rot_matrix[0,0]*100+rot_matrix[0,1]*100+rot_matrix[0,2]), int(rot_matrix[1,0]*100+rot_matrix[1,1]*100+rot_matrix[1,2]))
spot12 = (int(rot_matrix[0,0]*200+rot_matrix[0,1]*100+rot_matrix[0,2]), int(rot_matrix[1,0]*200+rot_matrix[1,1]*100+rot_matrix[1,2]))
spot21 = (int(rot_matrix[0,0]*100+rot_matrix[0,1]*300+rot_matrix[0,2]), int(rot_matrix[1,0]*100+rot_matrix[1,1]*300+rot_matrix[1,2]))
spot22 = (int(rot_matrix[0,0]*200+rot_matrix[0,1]*300+rot_matrix[0,2]), int(rot_matrix[1,0]*200+rot_matrix[1,1]*300+rot_matrix[1,2]))

lt = (spot21[0], spot11[1])
rb = (spot12[0], spot22[1])
pic2 = cv2.warpAffine(pic, rot_matrix, (pic.shape[1], pic.shape[0]))
cv2.circle(pic2, spot11, 4, (0, 255, 0), -1)
cv2.circle(pic2, spot12, 4, (0, 255, 0), -1)
cv2.circle(pic2, spot21, 4, (0, 255, 0), -1)
cv2.circle(pic2, spot22, 4, (0, 255, 0), -1)
cv2.rectangle(pic2,lt, rb, (0, 255, 0), 5)
cv2.imshow("win", pic)
cv2.waitKey(0)
"""
"""
road = "D:\VOC2012\images"
labels = "D:\VOC2012\labels"
os.makedirs(os.path.join(road, "images"), exist_ok=True)
filenames = os.listdir(labels)
for filename in filenames:
    shutil.copy(os.path.join(road, filename.split(".")[0]+".jpg"), os.path.join(road, "images", filename.split(".")[0]+".jpg"))
"""