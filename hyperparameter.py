import sys, os, cv2, pickle
from optparse import OptionParser
from torch.utils.data import DataLoader
from utils import build_channels, px_to_latlon
import torch
from models import MultiTaskSiamese
from dataset import HelioDataset
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


def estimate(siamese, device, num_workers):
    print('Starting hyperparameter tuning')

    val_dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                               '/homeRAID/efini/dataset/ground/validation',
                               '/homeRAID/efini/dataset/SDO/validation',
                               patch_size=200,
                               overlap=0.0)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=num_workers,
                                shuffle=True)

    tot_obs = 0
    ari = {e:0 for e in np.linspace(0.02,1,50)}
    for obs_idx, obs in enumerate(val_dataloader):
        try:
            if obs == 0:
                continue

            disk = np.array(obs['full_disk'][0])
            mask = np.array(obs['full_disk_mask'][0])
            instances = np.array(obs['full_disk_instances'][0])
            print(obs['date'])

            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
            disk_area = mask.shape[0] * mask.shape[1] - stats[0][4]

            true_clusters = [int(instances[labels==i][0]) for i in range(n)]
            embeddings = []

            for idx in range(1,n):
                features = build_channels(img=disk,
                                          stats=stats[idx],
                                          center=centers[idx],
                                          disk_area=disk_area,
                                          output_size=(100,100))
                e = siamese.embed(torch.stack([features]).float().to(device))
                # embeddings.append(px_to_latlon(centers[idx], 2000))
                embeddings.append(e.squeeze().detach().numpy())

            # predict
            for eps in np.linspace(0.02,1,50):
                pred_clusters = DBSCAN(eps=eps, min_samples=0).fit_predict(embeddings)
                ari[eps] += adjusted_rand_score(pred_clusters, true_clusters[1:])
                # print(np.amax(pred_clusters), len(set(true_clusters[1:])))
                # print("ARI:", adjusted_rand_score(pred_clusters, true_clusters[1:]))

            # plt.plot(ari.keys(), ari.values())
            # plt.show()

        except Exception as e:
            print('Error in validation')
            print(e)
            continue
        tot_obs += 1

    ari.update({k:v/tot_obs for k,v in ari.items()})
    pickle.dump(ari,open('/homeRAID/efini/logs/ari.pkl',"wb"))

    return




def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-w', '--num-workers', dest='num_workers', default=1,
                      type='int', help='number of workers')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    siamese = MultiTaskSiamese()

    if args.load:
        siamese.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    siamese = siamese.to(device)
    siamese.eval()

    estimate(siamese, device, args.num_workers)
