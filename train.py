import sys, os, cv2, random
from optparse import OptionParser
from torch.utils.data import DataLoader

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval
from models import UNet
from models import MultiTaskHybridSiamese

from dataset import HelioDataset
from loss import dice_coeff
from utils.utils import sample_sunspot_pairs

import tensorboardX
import numpy as np
from PIL import Image
from datetime import datetime

def train(unet,
          siamese,
          logdir,
          device,
          epochs=5,
          lr=0.001,
          save_cp=True,
          patch_size=200,
          sampling_ratio=0.2,
          num_workers=3,
          num_anchors=3):

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

            # log
            step = (epoch-1) * len(dataset) + obs_idx
            pred_masks = (pred_masks > 0.5).float()
            dice = dice_coeff(pred_masks, true_masks).item()
            writer.add_scalar('train/unet/bce-loss', bce_loss.item(), step)
            writer.add_scalar('train/unet/dice-coeff', dice, step)
            print('Observation', obs['date'][0], '| loss:',
                  '> bce: {:.6f} > dice {:.6f}'.format(bce_loss.item(), dice))

            # --- TRAIN SIAMESE --- #
            features = obs['sunspot_features'][0].float()
            clusters = obs['sunspot_clusters'][0].float()
            classes = obs['sunspot_classes'][0].float()
            print(clusters.shape)
            input, gt = sample_sunspot_pairs(features, clusters, classes,
                                             num_anchors=num_anchors)
            anchors, others = [i.to(device) for i in input]
            gt_sim, _, gt_class_other = [o.to(device) for o in gt]
            pred_sim, _, pred_class_other = siamese(anchors, others)
            sim_loss = bce(pred_sim, gt_sim)
            class_loss = bce(pred_class_other, gt_class_other.squeeze())
            loss = sim_loss + class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




            # log
            step = (epoch-1) * len(dataset) + obs_idx
            pred_masks = (pred_masks > 0.5).float()
            dice = dice_coeff(pred_masks, true_masks).item()
            writer.add_scalar('train/siamese/sim-loss', sim_loss.item(), step)
            writer.add_scalar('train/siamese/class-loss', class_loss.item(), step)
            print('> similarity: {:.6f} > classification {:.6f}'
                  .format(sim_loss.item(), class_loss.item()))


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
    parser.add_option('-w', '--num-workers', dest='num_workers', default=1,
                      type='int', help='number of workers')
    parser.add_option('-s', '--sampling-ratio', dest='sampling_ratio',
                      default=0.2, type='float', help='sampling ratio')


    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    unet = UNet(n_channels=1, n_classes=1)
    siamese = MultiTaskHybridSiamese()

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
