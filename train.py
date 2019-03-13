import sys
import os
from optparse import OptionParser
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval
from unet import UNet

from dataset import HelioDataset
from dice_loss import dice_coeff

import tensorboardX
from datetime import datetime

def train(net,
          logdir,
          device,
          epochs=5,
          lr=0.01,
          save_cp=True,
          epoch_size=10,
          patch_size=200,
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

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    bce = nn.BCELoss()

    for epoch in range(1,epochs+1):

        if 1:
            eval(net,
                 device,
                 patch_size=patch_size,
                 num_workers=num_workers,
                 epoch=epoch,
                 writer=writer,
                 num_viz=3)

        print('Starting epoch {}/{}.'.format(epoch, epochs))
        for obs_idx, obs in enumerate(dataloader):
            net.train()
            obs_loss = {'bce': 0, 'dice': 0}
            patches = obs['patches'][0].float().to(device)
            true_masks = obs['masks'][0].float().to(device)
            pred_masks = net(patches)
            pred_masks_flat = pred_masks.view(-1)
            true_masks_flat = true_masks.view(-1)
            bce_loss = bce(pred_masks_flat, true_masks_flat)
            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()

            obs_loss['bce'] += bce_loss.item()
            pred_masks = (pred_masks > 0.5).float()
            obs_loss['dice'] += dice_coeff(pred_masks, true_masks).item()
            global_step = (epoch-1)*len(dataset) + obs_idx
            writer.add_scalar('train-bce-loss', obs_loss['bce'], global_step)
            writer.add_scalar('train-dice-coeff', obs_loss['dice'], global_step)
            print('Observation', obs['date'][0], '| loss:',
                  *['> {}: {:.6f}'.format(k,v) for k,v in obs_loss.items()])

        print('Epoch finished!')

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP512-{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-z', '--epoch-size', dest='epoch', type='int',
                      default=10, help=' of the epochs')
    parser.add_option('-p', '--patch-size', type='int', dest='patch_size',
                      default=200, help="size of each patch of the images")
    parser.add_option('-d', '--logdir', type='str', dest='logdir',
                      default='/homeRAID/efini/logs', help="log directory")
    parser.add_option('-w', '--num-workers', dest='num_workers', default=3,
                      type='int', help='number of workers')



    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device(cuda)
        net = net.to(device)
    else:
        device = torch.device('cpu')

    try:
        train(net=net,
              logdir=args.logdir,
              epochs=args.epochs,
              lr=args.lr,
              epoch_size=args.epoch,
              patch_size=args.patch_size,
              num_workers=args.num_workers,
              device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
