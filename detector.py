import math
import torch as t
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from models.model import FcosDetector
from datasets.myDataset import DetectDataset
from config import Config
from utils.functions import *
def detect(model, imgs, size, roads, colors, classname):
    with t.no_grad():
        detections = model(imgs)
    batch = len(detections)

    results = []
    for i in range(batch):
        pic = cv2.imread(roads[i])
        detection = detections[i]
        if detection is not None:
            detection = clip_box(detection, size).cpu().detach().numpy()
            detection = cal_box(detection, pic.shape, size)
            #print(detection)
            for box in detection:
                x1, y1, x2, y2, cls, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]),  int(box[4]),round(float(box[-1]), 2)
                cv2.rectangle(pic, (x1, y1), (x2, y2), colors[cls], 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pic, classname[cls]+":"+str(score), (x1+5, y1+25), font, 0.8, colors[cls], 2)
            cv2.imshow("win", pic)
            cv2.waitKey(0)
        results.append(pic)
    return results

if __name__ == "__main__":
    config = Config()
    os.makedirs(config.detectResults, exist_ok=True)
    datasets = DetectDataset(config.detetcDirs, config.size)
    dataloader = DataLoader(datasets, batch_size=1, collate_fn=datasets.collate_fn, drop_last=False)
    model = FcosDetector(classNum=config.classNum)

    if t.cuda.is_available():
        model = model.cuda()
        model = t.nn.DataParallel(model, device_ids=[0, 1])

    model.load_state_dict(t.load(config.detectWeights))
    model.eval()

    for imgs, roads in dataloader:
        if t.cuda.is_available():
            imgs = imgs.cuda()
        results = detect(model, imgs, config.size, roads, config.colors, config.classname)
        for i, pic in enumerate(results):
           filename = roads[i].split("/")[-1]
           cv2.imwrite(config.detectResults + filename, pic)