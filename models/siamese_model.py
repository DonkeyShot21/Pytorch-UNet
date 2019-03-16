from torch import nn
from torch import FloatTensor
import torch
import numpy as np


class MultiTaskHybridSiamese(nn.Module):
    def __init__(self):
        super(MultiTaskHybridSiamese, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(5, 8, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc_sim = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),

            nn.Linear(200, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 1)
        )

        self.fc_class = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),

            nn.Linear(200, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 8)
        )

    def embed(self, input):
        out = self.cnn(input)
        return out.view(out.size()[0], -1)

    def similarity(self, e1, e2):
        out = self.fc_sim(e1 - e2)
        return torch.sigmoid(out)

    def classify(self, e):
        out = self.fc_class(e)
        return torch.sigmoid(out)

    def forward(self, in1, in2):
        e1 = self.embed(in1)
        e2 = self.embed(in2)
        sim = self.similarity(e1, e2)
        c1 = self.classify(e1)
        c2 = self.classify(e2)
        return sim, c1, c2


if __name__ == '__main__':
    in1 = torch.rand(1,5,100,100)
    in2 = torch.rand(1,5,100,100)
    sh = MultiTaskHybridSiamese()
    print(sh.forward(in1,in2))
