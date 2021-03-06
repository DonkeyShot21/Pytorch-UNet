import sys
import os
from optparse import OptionParser
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet

from dataset import HelioDataset
from dice_loss import dice_coeff

from utils.utils import keep_best, slice

# from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              epoch_size=10,
              window=512,
              obs_size=10):

    print('''Starting training:
                Epochs: {}
                Batch : {}
                Learning rate: {}
                Checkpoints: {}
                CUDA: {}
            '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    dir_checkpoint = 'checkpoints/'

    dataset = HelioDataset('data/SIDC_dataset.csv',
                           'data/fenyi',
                           epoch_size)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    bce = nn.BCELoss()

    data_loader = DataLoader(dataset)

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        for _, obs in enumerate(data_loader):
            obs = keep_best(slice(obs, window, window // 2), obs_size)

            for idx in range(0, len(obs['imgs']), batch_size):
                imgs = obs['imgs'][idx:idx+batch_size].float()
                true_masks = obs['masks'][idx:idx+batch_size].float()

                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)
                masks_probs_flat = masks_pred.view(-1)

                true_masks_flat = true_masks.view(-1)

                loss = bce(masks_probs_flat, true_masks_flat)

                print(int(idx),'-',int(idx+batch_size),'> loss: {0:.6f}'.format(loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Epoch finished!')

        #if 1:
        #    val_dice = eval_net(net, val, gpu)
        #    print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP512-{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch', default=5,
                      type='int', help='batch ')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-z', '--epoch-size', dest='epoch', type='int',
                      default=10, help=' of the epochs')
    parser.add_option('-w', '--window', type='int', dest='window', default=512,
                      help="size of each window/slice of the images")
    parser.add_option('-y', '--obs-size', dest='obs_size', default=10)

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=2, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch,
                  lr=args.lr,
                  gpu=args.gpu,
                  epoch_size=args.epoch,
                  window=args.window,
                  obs_size=args.obs_size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
