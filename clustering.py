from dataset import HelioDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch import FloatTensor
from torch.optim import Adam

import numpy as np
import cv2, pickle, random

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from utils import plot_mask, to_uint8


class NeuralSimilarity(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralSimilarity, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.sigmoid(out)

    def distance(self, x1, x2):
        x = FloatTensor([*x1, *x2])
        return 1 - self.forward(x)


if __name__ == "__main__":

    dataset = HelioDataset('data/sidc/SIDC_dataset.csv', 'data/dpd/', 1)
    data_loader = DataLoader(dataset)

    training_set = {"similarity": [], "eps_estimation": []}

    for _, obs in enumerate(data_loader):
        intensitygram = np.array(obs['inputs'][0][0])
        magnetogram = np.array(obs['inputs'][0][1])
        mask = np.array(obs['mask'][0], dtype=np.uint8)
        instances =  np.array(obs['instances'][0], dtype=np.uint8)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        min_intesitygram = [np.amin(intensitygram[labels==i]) for i in range(ret)]
        avg_magnetogram = [np.mean(magnetogram[labels==i]) for i in range(ret)]
        true_clusters = [int(instances[labels==i][0]) for i in range(ret)]

        obs_data = []
        for i in range(ret):
            row = [*centroids[i]/labels.shape[0], *stats[i]/labels.shape[0]]
            row.extend([min_intesitygram[i], avg_magnetogram[i]])
            obs_data.append(row)
        obs_data = np.array(obs_data, dtype=np.float32)
        training_set.append(obs_data)

        for i in range(1,ret):
            # one positive example and one negative for every sunspot
            c_id = true_clusters[i]
            same = [s for s in range(1,ret) if true_clusters[s] == c_id]
            other = [s for s in range(1,ret) if true_clusters[s] != c_id]
            positive = random.sample(same, 1)[0]
            negative = random.sample(other, 1)[0]

            # input = []
            # for j in [positive, negative]:
            #     distance = centroids[i] - centroids[j]
            #     intensity_diff = avg_intesitygram[i] - avg_intesitygram[j]
            #     magnetic_diff = avg_magnetogram[i] - avg_magnetogram[j]
            #     size_diff = stats[i][-1] - stats[i][-1]
            #     row = [*distance, intensity_diff, magnetic_diff, size_diff]
            #     input.append(row)

            input = [[*data[i], *data[e]] for e in [negative, positive]]
            gt = [[c_id == true_clusters[e]] for e in [negative, positive]]


    # instantiate my neural similarity
    net = NeuralSimilarity(2*data.shape[1], 20, 1)
    # define my loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(net.parameters(), lr=0.001)

    # train
    for epoch in range(0,10):
        print("epoch",epoch)
        epoch_loss = []
        for i in range(1,ret):
            # one positive example and one negative for every sunspot
            c_id = true_clusters[i]
            same = [s for s in range(1,ret) if true_clusters[s] == c_id]
            other = [s for s in range(1,ret) if true_clusters[s] != c_id]
            positive = random.sample(same, 1)[0]
            negative = random.sample(other, 1)[0]

            # input = []
            # for j in [positive, negative]:
            #     distance = centroids[i] - centroids[j]
            #     intensity_diff = avg_intesitygram[i] - avg_intesitygram[j]
            #     magnetic_diff = avg_magnetogram[i] - avg_magnetogram[j]
            #     size_diff = stats[i][-1] - stats[i][-1]
            #     row = [*distance, intensity_diff, magnetic_diff, size_diff]
            #     input.append(row)

            input = [[*data[i], *data[e]] for e in [negative, positive]]
            gt = [[c_id == true_clusters[e]] for e in [negative, positive]]

            pred = net(FloatTensor(input))
            loss = criterion(pred, FloatTensor(gt))

            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("average loss", np.mean(epoch_loss))

    # predict
    for eps in np.linspace(0.0001,0.2,20):
        pred_clusters = DBSCAN(eps=eps, metric=net.distance).fit_predict(data[1:])

        print("eps:",eps, np.amax(pred_clusters), np.amax(true_clusters))
        print("ARI:", adjusted_rand_score(pred_clusters, true_clusters[1:]))
