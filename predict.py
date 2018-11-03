import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms

from dataset import HelioDataset
from torch.utils.data import DataLoader

def predict_img(net,
                img,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=False):

    net.eval()

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        output_img = net(img)
        output_probs = F.sigmoid(output_img).squeeze(0)
        output_probs = output_probs.cpu().numpy()[0]

    if use_dense_crf:
        output_probs = dense_crf(np.array(output_probs).astype(np.uint8), output_probs)

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
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()



def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

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

    dataset = HelioDataset('./data/SIDC_dataset.csv',
                           'data/sDPD2014.txt',
                           1)
    data_loader = DataLoader(dataset)


    for _, obs in enumerate(data_loader):
        for idx in range(0, len(obs['imgs'][0])):
            print("\nPredicting image {} ...".format(idx))

            img = obs['imgs'][0][idx:idx+1].float()
            true_mask = obs['masks'][0][idx:idx+1].float()

            mask = predict_img(net=net,
                               img=img,
                               out_threshold=args.mask_threshold,
                               use_dense_crf= args.crf,
                               use_gpu= not args.cpu)

            if args.viz:
                print("Visualizing results for image {}, close to continue ...".format(fn))
                plot_img_and_mask(img, mask)

            if not args.no_save:
                out_fn = "results/predict" + str(idx) + ".bmp"
                result = mask_to_image(mask)
                result.save(out_fn)
                print("Mask saved to {}".format(out_fn))

                out_fn = "results/img" + str(idx) + ".bmp"
                img = mask_to_image(img.cpu().numpy()[0][0])
                img.save(out_fn)

                out_fn = "results/true_mask" + str(idx) + ".bmp"
                true_mask = mask_to_image(true_mask.cpu().numpy()[0][0])
                true_mask.save(out_fn)
