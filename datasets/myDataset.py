import torch as t
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import os
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
class TrainDataset(Dataset):
    def __init__(self, trainDirs, size=(800, 1333), flip=True, colorfilter=True):
        self.trainDirs = trainDirs
        self.labelsDirs = trainDirs.replace("images", "labels")
        filenames = os.listdir(self.trainDirs)
        self.imgsPath = [os.path.join(self.trainDirs, pic) for pic in filenames]
        self.labelsPath = [os.path.join(self.labelsDirs, label.replace("jpg", "txt").replace("png", "txt").replace("jpeg", "txt")) \
                           for label in filenames]
        self.size = size
    def __getitem__(self, item):
        imgroad = self.imgsPath[item]
        img = Image.open(imgroad)
        #img = bgr2rgb(img)
        labelroad = self.labelsPath[item]
        label = np.astype(np.loadtxt(labelroad), "float32")


    def __len__(self):
        pass

if __name__ == "__main__":
    pass