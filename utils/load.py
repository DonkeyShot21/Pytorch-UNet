#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw
from sunpy.net.vso import VSOClient


# FENNEC'S loading functions

def search_VSO(start_time, end_time):
    client = VSOClient()
    query_response = client.query_legacy(tstart=start_time,
                                         tend=end_time,
                                         instrument='HMI',
                                         physobs='intensity',
                                         sample=3600)
    results = client.fetch(query_response[:1],
                           path='./tmp/{file}',
                           site='rob')
    continuum_file = results.wait()

    query_response = client.query_legacy(tstart=start_time,
                                         tend=end_time,
                                         instrument='HMI',
                                         physobs='los_magnetic_field',
                                         sample=3600)
    results = client.fetch(query_response[:1],
                           path='./tmp/{file}',
                           site='rob')
    magnetic_file = results.wait()
    return continuum_file[0], magnetic_file[0]

def normalize_map(map):
    img = map.data
    img[np.isnan(img)] = 0
    img_min = np.amin(img)
    img_max = np.amax(img)
    return (img - img_min) / (img_max - img_min)

def remove_if_exists(file):
    if file != None:
        if os.path.exists(file):
            os.remove(file)

# -----------------------------------------------------------------------------



def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
