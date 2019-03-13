import sys
import os
from optparse import OptionParser
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils

from eval import eval
from unet import UNet

from dataset import HelioDataset
from dice_loss import dice_coeff

import tensorboardX
from datetime import datetime

def train(net,
          logdir,
          epochs=5,
          batch_size=5,
          lr=0.01,
          save_cp=True,
          gpu=False,
          epoch_size=10,
          patch_size=200,
          obs_size=1):

    print('''Starting training:
                Epochs: {}
                Batch : {}
                Learning rate: {}
                Checkpoints: {}
                CUDA: {}
          '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    dir_checkpoint = '/homeRAID/efini/checkpoints/'
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = tensorboardX.SummaryWriter(os.path.join(logdir, now))



    dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                           '/homeRAID/efini/dataset/ground/train',
                           '/homeRAID/efini/dataset/SDO/train')
    dataloader = DataLoader(dataset,
                            batch_size=obs_size,
                            shuffle=True)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    bce = nn.BCELoss()

    for epoch in range(1,epochs+1):
        print('Starting epoch {}/{}.'.format(epoch, epochs))
        for obs_idx, obs in enumerate(dataloader):
            net.train()
            obs_loss = {'bce': 0, 'dice': 0}
            num_patches = len(obs['patches'])
            for idx in range(0, num_patches, batch_size):
                patches = obs['patches'][idx:idx+batch_size].float()[0]
                true_masks = obs['masks'][idx:idx+batch_size].float()[0]
                if gpu:
                    patches = patches.cuda()
                    true_masks = true_masks.cuda()
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

            global_step = epoch*len(dataset) + obs_idx
            obs_loss.update((k,v/num_patches) for k,v in obs_loss.items())
            writer.add_scalar('train-bce-loss', obs_loss['bce'], global_step)
            writer.add_scalar('train-dice-coeff', obs_loss['dice'], global_step)
            print('Observation', obs['date'][0], '| validation loss:',
                  *['> {}: {:.6f}'.format(k,v) for k,v in obs_loss.items()])

        print('Epoch finished!')

        if 1:
           val_loss, val_plots = eval(net, batch_size, gpu, num_viz=3)
           writer.add_scalar('val-bce-loss', val_loss['bce'], epoch)
           writer.add_scalar('val-dice-coeff', val_loss['dice'], epoch)
           val_plots = val_plots.permute(0,3,1,2)
           grid = vutils.make_grid(val_plots, normalize=True)
           writer.add_image('val-viz', grid, epoch)
           print('Average validation loss:',
                 *['> {}: {:.6f}'.format(k,v) for k,v in val_loss.items()])


        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP512-{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch', default=5,
                      type='int', help='batch size')
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
    parser.add_option('-w', '--patch_size', type='int', dest='patch_size',
                      default=512, help="size of each patch of the images")
    parser.add_option('-d', '--logdir', type='str', dest='logdir',
                      default='/homeRAID/efini/logs', help="log directory")
    parser.add_option('-y', '--obs-size', dest='obs_size', default=1)

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train(net=net,
              logdir=args.logdir,
              epochs=args.epochs,
              batch_size=args.batch,
              lr=args.lr,
              gpu=args.gpu,
              epoch_size=args.epoch,
              patch_size=args.patch_size,
              obs_size=args.obs_size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
