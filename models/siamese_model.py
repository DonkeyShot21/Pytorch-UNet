from torch import nn
from torch import FloatTensor
import torch
import numpy as np


class SiameseHybrid(nn.Module):
    def __init__(self):
        super(SiameseHybrid, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),

            nn.Linear(200, 200),
            nn.ReLU(inplace=True),

            nn.Linear(200, 8)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 1)
        )

    def extract_features(self, x):
        output = self.cnn(x)
        return output.view(output.size()[0], -1)

    def similarity(self, fc1_in, distance):
        fc1_out = self.fc1(fc1_in)
        fc2_out = self.fc2(torch.cat((fc1_out, distance), 1))
        return torch.sigmoid(fc2_out)

    def forward(self, in1, in2, distance):
        sim_in = self.extract_features(in1) - self.extract_features(in2)
        return self.similarity(sim_in, distance)


if __name__ == '__main__':
    in1 = FloatTensor(np.ones((1,1,100,100), dtype=np.float32))
    in2 = FloatTensor(np.ones((1,1,100,100), dtype=np.float32))
    distance = FloatTensor(np.ones((1,2)))
    sh = SiameseHybrid()
    print(sh.forward(in1,in2,distance))
