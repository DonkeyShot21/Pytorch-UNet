import torch
from torch.nn import BCELoss
import torch.nn.functional as F
from dataset import HelioDataset
from torch.utils.data import DataLoader
from dice_loss import dice_coeff
from utils import plot_mask, to_uint8
import torchvision.utils as vutils
from torch import Tensor
import numpy as np
from PIL import Image




def eval(net, batch_size, gpu=False, num_viz=3):
    print('Starting validation')
    net.eval()
    bce = BCELoss()

    val_dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                               '/homeRAID/efini/dataset/ground/validation',
                               '/homeRAID/efini/dataset/SDO/validation')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=True)

    viz = []
    val_loss = {'bce': 0, 'dice': 0}
    for obs_idx, obs in enumerate(val_dataloader):
        obs_loss = {'bce': 0, 'dice': 0}
        num_patches = len(obs['patches'][0])
        for idx in range(0, num_patches, batch_size):
            patches = obs['patches'][0][idx:idx+batch_size].float()
            true_masks = obs['masks'][0][idx:idx+batch_size].float()
            if gpu:
                patches = patches.cuda()
                true_masks = true_masks.cuda()
            pred_masks = net(patches)
            pred_masks_flat = pred_masks.view(-1)
            true_masks_flat = true_masks.view(-1)
            bce_loss = bce(pred_masks_flat, true_masks_flat)
            obs_loss['bce'] += bce_loss.item()
            pred_masks = (pred_masks > 0.5).float()
            obs_loss['dice'] += dice_coeff(pred_masks, true_masks).item()
            if obs_idx < num_viz:
                viz_patches = to_uint8(np.array(patches))
                viz_true =np.array(true_masks)
                viz_pred = np.array(pred_masks)#.astype(np.uint8)
                plots = np.array([[plot_mask(viz_patches[i][0], viz_pred[i][0]),
                                   plot_mask(viz_patches[i][0], viz_true[i][0])]
                                   for i in range(len(patches))])
                viz.extend(plots.reshape(2*len(patches),*plots.shape[2:]))

        obs_loss['bce'] /= num_patches
        obs_loss['dice'] /= num_patches
        print('Observation', obs['date'][0], '| validation loss:',
              *['> {}: {:.6f}'.format(k,v) for k,v in obs_loss.items()])
        val_loss['bce'] += obs_loss['bce']
        val_loss['dice'] +=  obs_loss['dice']

    return {k:v/len(val_dataset) for k,v in val_loss.items()}, Tensor(viz)




    #
    # for i, b in enumerate(dataset):
    #     img = b[0]
    #     true_mask = b[1]
    #
    #     img = torch.from_numpy(img).unsqueeze(0)
    #     true_mask = torch.from_numpy(true_mask).unsqueeze(0)
    #
    #     if gpu:
    #         img = img.cuda()
    #         true_mask = true_mask.cuda()
    #
    #     mask_pred = net(img)[0]
    #     mask_pred = (mask_pred > 0.5).float()
    #
    #     tot += dice_coeff(mask_pred, true_mask).item()
