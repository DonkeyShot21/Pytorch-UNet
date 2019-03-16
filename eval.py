import torch
from torch.nn import BCELoss
import torch.nn.functional as F
from dataset import HelioDataset
from torch.utils.data import DataLoader
from loss import dice_coeff
from utils import plot_mask, to_uint8
import torchvision.utils as vutils
from torch import Tensor
import numpy as np
from PIL import Image




def eval(net, device, patch_size, num_workers, writer, epoch,
         gpu=False, num_viz=3):
    print('Starting validation')
    net.eval()
    bce = BCELoss()

    val_dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                               '/homeRAID/efini/dataset/ground/validation',
                               '/homeRAID/efini/dataset/SDO/validation',
                               patch_size=patch_size)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=num_workers,
                                shuffle=True)

    viz = []
    val_loss = {'bce': 0, 'dice': 0}
    for obs_idx, obs in enumerate(val_dataloader):
        obs_loss = {'bce': 0, 'dice': 0}
        patches = obs['patches'][0].float().to(device)
        true_masks = obs['masks'][0]. float().to(device)
        pred_masks = net(patches)
        pred_masks_flat = pred_masks.view(-1)
        true_masks_flat = true_masks.view(-1)
        bce_loss = bce(pred_masks_flat, true_masks_flat)
        obs_loss['bce'] += bce_loss.item()
        pred_masks = (pred_masks > 0.5).float()
        obs_loss['dice'] += dice_coeff(pred_masks, true_masks).item()

        if obs_idx < num_viz:
            patches_np = to_uint8(np.array(patches.cpu()))
            true_np = np.array(true_masks.cpu())
            pred_np = np.array(pred_masks.cpu())
            plots = np.array([[plot_mask(patches_np[i][0], pred_np[i][0]),
                               plot_mask(patches_np[i][0], true_np[i][0])]
                               for i in range(len(patches))])
            viz.extend(plots.reshape(2*len(patches),*plots.shape[2:]))

        print('Observation', obs['date'][0], '| validation loss:',
              *['> {}: {:.6f}'.format(k,v) for k,v in obs_loss.items()])
        val_loss['bce'] += obs_loss['bce']
        val_loss['dice'] +=  obs_loss['dice']

    val_loss.update({k:v/len(val_dataset) for k,v in val_loss.items()})
    writer.add_scalar('val/bce-loss', val_loss['bce'], epoch)
    writer.add_scalar('val/dice-coeff', val_loss['dice'], epoch)
    viz = Tensor(viz).permute(0,3,1,2)
    grid = vutils.make_grid(viz, normalize=True)
    writer.add_image('val-viz', grid, epoch)

    print('Average validation loss:',
         *['> {}: {:.6f}'.format(k,v) for k,v in val_loss.items()])

    return




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
