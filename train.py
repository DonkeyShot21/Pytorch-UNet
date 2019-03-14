import sys, os, cv2, random
from optparse import OptionParser
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval
from models import UNet
from models import SiameseHybrid

from dataset import HelioDataset
from loss import dice_coeff

import tensorboardX
from datetime import datetime

from PIL import Image

def train(unet,
          siamese,
          logdir,
          device,
          epochs=5,
          lr=0.001,
          save_cp=True,
          patch_size=200,
          sampling_ratio=0.2,
          num_workers=3):

    dir_checkpoint = '/homeRAID/efini/checkpoints/'
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = tensorboardX.SummaryWriter(os.path.join(logdir, now))

    dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                           '/homeRAID/efini/dataset/ground/train',
                           '/homeRAID/efini/dataset/SDO/train',
                           patch_size=patch_size)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            shuffle=True)

    optimizer = optim.SGD(unet.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    bce = nn.BCELoss()

    for epoch in range(1,epochs+1):
        unet.train()
        siamese.train()
        print('Starting epoch {}/{}.'.format(epoch, epochs))
        for obs_idx, obs in enumerate(dataloader):

            # --- TRAIN UNET --- #
            patches = obs['patches'][0].float().to(device)
            true_masks = obs['masks'][0].float().to(device)
            pred_masks = unet(patches)
            pred_masks_flat = pred_masks.view(-1)
            true_masks_flat = true_masks.view(-1)
            bce_loss = bce(pred_masks_flat, true_masks_flat)
            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            # log unet stuff
            step = (epoch-1) * len(dataset) + obs_idx
            pred_masks = (pred_masks > 0.5).float()
            dice = dice_coeff(pred_masks, true_masks).item()
            writer.add_scalar('train-unet-bce-loss', bce_loss.item(), step)
            writer.add_scalar('train-unet-dice-coeff', dice, step)
            print('Observation', obs['date'][0], '| loss:',
                  '> bce: {:.6f} > dice {:.6f}'.format(bce_loss.item(), dice))

            # --- TRAIN SIAMESE --- #
            full_disk = obs['full_disk'][0].float().to(device)
            instances = np.array(obs['full_disk_instances'][0])
            mask = np.array(obs['full_disk_mask'][0], dtype=np.uint8)

            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
            true_clusters = [int(instances[labels==i][0]) for i in range(n)]
            print(instances.dtype, np.amin(instances), np.amax(instances))


            for i in range(1, n):
                print(labels[i], stats[i], centers[i])

            anchors = random.sample(range(1, 100), int(n * sampling_ratio) + 1)
            for anchor in anchors:
                # one positive example and one negative for every sunspot
                c_id = true_clusters[i]
                same = [s for s in range(1,ret) if true_clusters[s] == c_id]
                other = [s for s in range(1,ret) if true_clusters[s] != c_id]
                positive = random.sample(same, 1)[0]
                negative = random.sample(other, 1)[0]

                # input = []
                # for j in [positive, negative]:
                #     distance = centers[i] - centers[j]
                #     intensity_diff = avg_intesitygram[i] - avg_intesitygram[j]
                #     magnetic_diff = avg_magnetogram[i] - avg_magnetogram[j]
                #     size_diff = stats[i][-1] - stats[i][-1]
                #     row = [*distance, intensity_diff, magnetic_diff, size_diff]
                #     input.append(row)

                input = [[*data[i], *data[e]] for e in [negative, positive]]
                gt = [[c_id == true_clusters[e]] for e in [negative, positive]]



            siamese()


        print('Epoch finished!')

        if 1:
            eval(net,
                 device,
                 patch_size=patch_size,
                 num_workers=num_workers,
                 epoch=epoch,
                 writer=writer,
                 num_viz=3)


        if save_cp:
            torch.save(unet.state_dict(),
                       dir_checkpoint + 'CP512-{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-z', '--epoch-size', dest='epoch', type='int',
                      default=10, help=' of the epochs')
    parser.add_option('-p', '--patch-size', type='int', dest='patch_size',
                      default=200, help="size of each patch of the images")
    parser.add_option('-d', '--logdir', type='str', dest='logdir',
                      default='/homeRAID/efini/logs', help="log directory")
    parser.add_option('-w', '--num-workers', dest='num_workers', default=3,
                      type='int', help='number of workers')
    parser.add_option('-s', '--sampling-ratio', dest='sampling_ratio',
                      default=0.2, type='float', help='sampling ratio')


    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    unet = UNet(n_channels=1, n_classes=1)
    siamese = SiameseHybrid()

    if args.load:
        unet.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    unet = unet.to(device)
    siamese = siamese.to(device)

    try:
        train(unet=unet,
              siamese=siamese,
              logdir=args.logdir,
              epochs=args.epochs,
              lr=args.lr,
              patch_size=args.patch_size,
              num_workers=args.num_workers,
              device=device)
    except KeyboardInterrupt:
        torch.save(unet.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
