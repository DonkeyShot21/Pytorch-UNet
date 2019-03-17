from torch import nn
from torch import FloatTensor
import torch
import numpy as np


class MultiTaskSiamese(nn.Module):
    def __init__(self):
        super(MultiTaskSiamese, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc_embed = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 5)
        )

        self.fc_class = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 8)
        )

    def forward_cnn(self, input):
        out = self.cnn(input)
        return out.view(out.size()[0], -1)

    def embed(self, input):
        out = self.forward_cnn(input)
        out = self.fc_embed(out)
        return out

    def classify(self, e):
        out = self.fc_class(e)
        return torch.sigmoid(out)

    def forward(self, in1, in2):
        out1 = self.forward_cnn(in1)
        out2 = self.forward_cnn(in2)
        c1 = self.classify(out1)
        c2 = self.classify(out2)
        e1 = self.fc_embed(out1)
        e2 = self.fc_embed(out2)
        return e1, e2, c1, c2


if __name__ == '__main__':
    in1 = torch.rand(1,5,100,100)
    in2 = torch.rand(1,5,100,100)
    sh = MultiTaskSiamese()
    print(sh.forward(in1,in2))
