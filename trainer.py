import torch as t
import numpy as np
import cv2
import os
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from models.model import FcosTrainer
from models.loss import loss
from datasets.myDataset import TrainDataset
from config import Config
def train():
    config = Config()
    os.makedirs(config.weightsSave, exist_ok=True)
    model = FcosTrainer(weights=config.preTrain, classNum=config.classNum)

    if t.cuda.is_available():
        print("----GPU-Training----")
        model = model.cuda()
        model = t.nn.DataParallel(model, device_ids=[0, 1])

    if config.trainweights is not None:
        print("trainWeights:", config.trainweights)
        model.load_state_dict(t.load(config.trainweights))
    #for name, param in model.named_parameters():
    #    print(name, param)
    model.train()
    optimer = SGD(model.parameters(), lr=config.start_lr)
    optimer.zero_grad()
    scheduler = lr_scheduler.MultiStepLR(optimer, config.lr_change, config.lr_decay)
    datasets = TrainDataset(trainDirs=config.imgsRoad, size=config.size)
    dataloader = DataLoader(datasets, batch_size=config.batch, shuffle=True, collate_fn=datasets.collate_fn, drop_last=True)
    Loss = loss()

    for epoch in range(config.epoches):
        print("epoch-{}".format(epoch))
        for i, (imgs, labels, imgsPath) in enumerate(dataloader):

            print("--epoch-{}-batch-{}--".format(epoch, i))
            if t.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs, targets = model(imgs, labels)

            all_loss = Loss(outputs, targets)
            print("Loss:", all_loss)
            all_loss.backward()

            optimer.step()
            optimer.zero_grad()
        scheduler.step()
        if (epoch+1) % 4 == 0:
            t.save(model.state_dict(), config.weightsSave + "epoch{}.pth".format(100+epoch))
    t.save(model.state_dict(), config.weightsSave + "finally.pth")
if __name__ == "__main__":
    train()