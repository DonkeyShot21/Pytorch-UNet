import random, torch
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import solar_rotate_coordinate

# FENNEC'S FUNCTIONS

def rotate_coord(map, coord, date):
    coord_sc = SkyCoord(
        [(float(v[1]),float(v[0])) * u.deg for v in np.array(coord)],
        obstime=date,
        frame=frames.HeliographicCarrington)
    coord_sc = coord_sc.transform_to(frames.Helioprojective)
    rotated_coord_sc = solar_rotate_coordinate(coord_sc, map.date)

    px = map.world_to_pixel(rotated_coord_sc)
    return [(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]

def patchify(full_disk, full_disk_mask, patch_size, overlap):
    patches = []
    masks = []
    stride = int(patch_size - (overlap * patch_size))
    for x in range(0, full_disk.shape[0]-patch_size+1, stride):
        for y in range(0, full_disk.shape[1]-patch_size+1, stride):
            patch = full_disk[x:x+patch_size,y:y+patch_size]
            mask = full_disk_mask[x:x+patch_size,y:y+patch_size]
            patches.append([patch])
            masks.append([mask])
    return np.array(patches), np.array(masks)


def sample_patches(patches, masks, num_patches):
    counts = np.array([np.count_nonzero(m) for m in masks])
    # if you want the best n patches use this:
    # indices = np.argpartition(counts, -n)[-n:]
    # instead we sample with probability:
    probs = counts / sum(counts)
    indices = np.random.choice(len(masks), num_patches, p=probs, replace=False)
    return patches[indices], masks[indices]

def normalize_map(map):
    img = map.data
    return normalize_img(img)

def normalize_img(img):
    img[np.isnan(img)] = 0
    img_min = np.amin(img)
    print(img_min)
    img_max = np.amax(img)
    print(img_max)
    return (img - img_min) / (img_max - img_min)

def to_uint8(img):
    return np.array(img * 255, dtype=np.uint8)

# ------------------------------------------------------------------------------


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
