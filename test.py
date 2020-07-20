import torch as t
import numpy as np
from torchvision import transforms
import numpy as np
import cv2
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
cv2.imshow("win", pic2)
cv2.waitKey(0)
