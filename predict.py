import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw
from utils import plot_img_and_mask, merge_masks#, dense_crf
from utils import slice, keep_best, plot_mask, to_uint8

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
        output = net(imgs).cpu().numpy()

    # if use_dense_crf:
    #     rgb = cv2.cvtColor(np.array(img[0][0] * 255, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
    #     output_probs = dense_crf(rgb, output_probs)

    return output > out_threshold



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


if __name__ == "__main__":
    args = get_args()

    net = UNet(n_channels=2, n_classes=1)

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

    dataset = HelioDataset('./data/sidc/SIDC_dataset.csv',
                           'data/dpd/',
                           1)
    data_loader = DataLoader(dataset)

    pred_mask_slices = []
    input = None
    true_mask = None

    for _, obs in enumerate(data_loader):
        input = np.array(obs["inputs"][0])
        true_mask = np.array(obs["mask"][0])
        obs = slice(obs, args.window, args.window)
        for idx in range(0, len(obs['inputs']), args.batch_size):
            print("\nPredicting images {0} - {1} ...".format(idx, idx+args.batch_size))
            imgs = obs['inputs'][idx:idx+args.batch_size].float()

            masks = predict_img(net=net,
                                imgs=imgs,
                                out_threshold=args.mask_threshold,
                                use_dense_crf= args.crf,
                                use_gpu= not args.cpu)

            pred_mask_slices.extend(masks.squeeze())

    pred_mask_slices = np.array(pred_mask_slices)
    n = input[0].shape[0] // args.window
    rows = [pred_mask_slices[i:i+n] for i in range(0,len(pred_mask_slices),n)]
    rows = np.array(rows)
    predicted_mask = np.vstack([np.hstack(a) for a in rows])

    if args.viz:
        print("Visualizing results for image {}, close to continue ...".format(fn))
        plot_mask(to_uint8(input[0]), mask).show()
        plot_mask(to_uint8(input[0]), true_mask).show()

    if not args.no_save:
        out_fn = "results/predicted.bmp"
        result = plot_mask(to_uint8(input[0]), predicted_mask)
        result.save(out_fn)
        print("Predicted mask saved to {}".format(out_fn))

        out_fn = "results/true_mask.bmp"
        true_mask = plot_mask(to_uint8(input[0]), true_mask)
        true_mask.save(out_fn)
        print("True mask saved to {}".format(out_fn))
