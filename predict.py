import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw
from utils import plot_img_and_mask, merge_masks, dense_crf
from utils import slice, keep_best, plot_mask

from torchvision import transforms

from dataset import HelioDataset
from torch.utils.data import DataLoader
import cv2

def predict_img(net,
                imgs,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=False):

    net.eval()

    if use_gpu:
        imgs = imgs.cuda()

    with torch.no_grad():
        output_imgs = net(imgs)
        output_probs = F.sigmoid(output_imgs)
        output_probs = output_probs.cpu().numpy()

    if use_dense_crf:
        rgb = cv2.cvtColor(np.array(img[0][0] * 255, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
        output_probs = dense_crf(rgb, output_probs)

    return output_probs > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--window', '-w', type=int,
                        help="Size of each window/slice of the images",
                        default=512)
    parser.add_argument('--batch-size', '-b', type=int,
                        help="Batch size",
                        default=16)

    return parser.parse_args()

def to_image(img):
    return Image.fromarray((img * 255).astype(np.uint8))

def to_uint8(img):
    return np.array(img * 255, dtype=np.uint8)

if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=2, n_classes=2)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    dataset = HelioDataset('./data/SIDC_dataset.csv',
                           'data/fenyi',
                           1)
    data_loader = DataLoader(dataset)

    pred_mask_slices = []
    cont_image = None
    true_mask = None

    for _, obs in enumerate(data_loader):
        cont_image = np.array(obs["img"][0][0])
        true_masks = np.array(obs["mask"][0])
        obs = slice(obs, args.window, args.window)
        for idx in range(0, len(obs['imgs']), args.batch_size):
            print("\nPredicting images {0} - {1} ...".format(idx, idx+args.batch_size))
            imgs = obs['imgs'][idx:idx+args.batch_size].float()

            masks = predict_img(net=net,
                                imgs=imgs,
                                out_threshold=args.mask_threshold,
                                use_dense_crf= args.crf,
                                use_gpu= not args.cpu)

            pred_mask_slices.extend(masks.squeeze())

    pred_mask_slices = np.array(pred_mask_slices)
    n = cont_image.shape[0] // args.window
    rows = [pred_mask_slices[i:i+n] for i in range(0,len(pred_mask_slices),n)]
    cols = np.array([np.concatenate(r,axis=2) for r in np.array(rows)])
    predicted_masks = np.concatenate(cols,axis=1)

    predicted_result = []
    true_mask_result = []
    for i in range(len(predicted_masks)):
        predicted_result.append(plot_mask(to_uint8(cont_image), predicted_masks[i]))
    for i in range(len(true_masks)):
        true_mask_result.append(plot_mask(to_uint8(cont_image), true_masks[i]))

    if args.viz:
        for i in range(len(predicted_result)):
            predicted_result[i].show()
        for i in range(len(true_mask_result)):
            true_mask_result[i].show()

    if not args.no_save:
        for i in range(len(predicted_result)):
            out_fn = "results/predicted{}.bmp".format(i)
            predicted_result[i].save(out_fn)
            print("Predicted mask saved to {}".format(out_fn))
        for i in range(len(true_mask_result)):
            out_fn = "results/true_mask{}.bmp".format(i)
            true_mask_result[i].save(out_fn)
            print("True mask saved to {}".format(out_fn))
