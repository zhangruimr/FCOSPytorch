import torch as t
import torch.nn as nn
import resnet
import math
def init(model):
    for name, param in model.named_parameters():
        if name.find("bias") >= 0:
            nn.init.constant_(param, 0)
        elif name.find("weight") >= 0 and name.find("GN") < 0:
            nn.init.normal_(param, 0, 0.01)
class Resnet(resnet.ResNet):

    #默认使用resnet101
    def __init__(self, block=resnet.Bottleneck, layers=[3, 4, 23, 3], weights=None):
        super(Resnet, self).__init__(block, layers)
        if not weights == None:
            self.load_state_dict(t.load(weights))
            print("-----加载預训练成功-----")
        del self.avgpool
        del self.fc

def Backbone(resnettype = 101, weights=None):

    block = resnet.Bottleneck
    layers = [3, 4, 23, 3]

    if resnettype == 101:
        model = Resnet(block, layers, weights)
        return model
    elif resnettype == 152:
        layers = [3, 8, 36, 3]
        model = Resnet(block, layers, weights)
        return model

class FPN(nn.Module):
    def __init__(self, inputChannels, outChannel=256):
        super(FPN, self).__init__()
        c3_channel, c4_channel, c5_channel = inputChannels

        self.p5_1 = nn.Conv2d(c5_channel, outChannel, (1, 1), 1, 0)
        self.p5_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p4_1 = nn.Conv2d(c4_channel, outChannel, (1, 1), 1, 0)
        self.p4_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p3_1 = nn.Conv2d(c3_channel, outChannel, (1, 1), 1, 0)
        self.p3_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_2 = nn.Conv2d(outChannel, outChannel, (3, 3), 1, 1)

        self.p6 = nn.Conv2d(outChannel, outChannel, (3, 3), 2, 1)

        self.p7_relu = nn.ReLU()
        self.p7_conv = nn.Conv2d(outChannel, outChannel, (3, 3), 2, 1)
        init(self)
    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_x = self.p5_1(c5)
        p5_upsample = self.p5_up(p5_x)
        p5_x = self.p5_2(p5_x)
        #print("p5", p5_x.shape)
        p4_x = self.p4_1(c4)
        p4_x = p4_x + p5_upsample
        p4_upsample = self.p4_up(p4_x)
        p4_x = self.p4_2(p4_x)
        #print("p4", p4_x.shape)
        p3_x = self.p3_1(c3)
        p3_x = p3_x + p4_upsample
        p3_x = self.p3_2(p3_x)
        #print("p3", p3_x.shape)
        p6_x = self.p6(p5_x)

        #print("p6", p6_x.shape)
        p7_x = self.p7_relu(p6_x)
        p7_x = self.p7_conv(p7_x)
        #print("p7", p7_x.shape)
        output = [p3_x, p4_x, p5_x, p6_x, p7_x]

        return output

class detectHead(nn.Module):
    def __init__(self, inchannels=256, classNum=90):
        super(detectHead, self).__init__()
        self.clsbranch = nn.Sequential()
        self.regbranch = nn.Sequential()

        for i in range(4):
            self.clsbranch.add_module("clsConv2d_{}".format(i), nn.Conv2d(inchannels, inchannels, 3, padding=1, bias=True))
            self.clsbranch.add_module("clsGN_{}".format(i), nn.GroupNorm(32, inchannels))
            self.clsbranch.add_module("clsRELU_{}".format(i), nn.ReLU(inplace=True))

            self.regbranch.add_module("regConv2d_{}".format(i), nn.Conv2d(inchannels, inchannels, 3, padding=1, bias=True))
            self.regbranch.add_module("regGN_{}".format(i), nn.GroupNorm(32, inchannels))
            self.regbranch.add_module("regRELU_{}".format(i), nn.ReLU(inplace=True))
        init(self.clsbranch)
        init(self.regbranch)

        self.clsHead = nn.Conv2d(inchannels, classNum, 3, padding=1)
        nn.init.normal_(self.clsHead.weight, 0, 0.01)
        nn.init.constant_(self.clsHead.bias, -math.log((1 - 0.1) / 0.1))
        self.clsSigmoid = nn.Sigmoid()

        self.centerHead = nn.Conv2d(inchannels, 1, 3, padding=1)
        nn.init.normal_(self.centerHead.weight, 0, 0.01)
        nn.init.constant_(self.centerHead.bias, -math.log((1 - 0.1) / 0.1))

        self.regHead = nn.Conv2d(inchannels, 4, 3, padding=1)
        nn.init.normal_(self.regHead.weight, 0, 0.01)
        nn.init.constant_(self.regHead.bias, -math.log((1 - 0.1) / 0.1))

    def forward(self, inputs):
        cls = []
        cnt = []
        reg = []
        for i, fpn_out in enumerate(inputs):
            out = self.clsbranch(fpn_out)

            clsout = self.clsHead(out)
            cls.append(self.clsSigmoid(clsout))

            cnt.append(self.centerHead(out))


            regout = self.regbranch(fpn_out)
            reg.append(self.regHead(regout))
        return cls, cnt, reg