import random, torch
import numpy as np
import math, cv2
from PIL import Image

from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import solar_rotate_coordinate


def rotate_coord(map, coord, date):
    coord_sc = SkyCoord(
        [(float(v[1]),float(v[0])) * u.deg for v in np.array(coord)],
        obstime=date,
        frame=frames.HeliographicCarrington)
    coord_sc = coord_sc.transform_to(frames.Helioprojective)
    rotated_coord_sc = solar_rotate_coordinate(coord_sc, map.date)

    px = map.world_to_pixel(rotated_coord_sc)
    return [(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]

def bbox(img):
    a = np.where(img != 0)
    bbox = [np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])]
    return bbox

def patchify(full_disk, full_disk_mask, patch_size, overlap):
    patches = []
    masks = []
    ph = patch_size // 2
    padded = np.pad(full_disk, ((ph,ph),(ph,ph)), mode='constant')
    padded_mask = np.pad(full_disk_mask, ((ph,ph),(ph,ph)), mode='constant')
    stride = int(patch_size - (overlap * patch_size))
    xmin, xmax, ymin, ymax = bbox(padded_mask)
    for x in range(xmin-ph, xmax+ph+1, stride):
        for y in range(ymin-ph, ymax-ph+1, stride):
            patch = padded[x:x+patch_size,y:y+patch_size]
            mask = padded_mask[x:x+patch_size,y:y+patch_size]
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
    img_max = np.amax(img)
    return (img - img_min) / (img_max - img_min)

def to_uint8(img):
    return np.array(img * 255, dtype=np.uint8)

def sample_sunspot_pairs(disk, mask, instances, classes, num_anchors):
    anchors, others = [], []
    similarity, anchor_classes, other_classes = [], [], []

    tmp = np.ma.masked_where(mask == 0, instances)
    if np.amin(tmp) > 254 or np.amax(mask) == 0:
        return [-1, -1], [-1, -1, -1]

    num_anchors = max(255 - np.amin(tmp), num_anchors)

    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    disk_area = mask.shape[0] * mask.shape[1] - stats[0][4]
    true_clusters = [int(instances[labels==i][0]) for i in range(n)]
    true_classes = np.array([int(classes[labels==i][0]) for i in range(n)])
    true_classes[true_classes == 0] = random.randrange(65,72)
    true_classes -= 65

    for _ in range(num_anchors):
        anchor = random.choice(range(1, n))
        c_id = true_clusters[anchor]
        same = [s for s in range(1,n) if true_clusters[s] == c_id]
        other = [o for o in range(1,n) if true_clusters[o] != c_id]
        positive_id = random.choice(same)
        negative_id = random.choice(other)
        anchor_features = build_channels(img=disk,
                                         stats=stats[anchor],
                                         center=centers[anchor],
                                         disk_area=disk_area,
                                         output_size=(100,100))

        for sim, other in enumerate([negative_id, positive_id]):
            anchors.append(anchor_features)
            others.append(build_channels(img=disk,
                                          stats=stats[other],
                                          center=centers[other],
                                          disk_area=disk_area,
                                          output_size=(100,100)))
            similarity.append([sim])
            anchor_classes.append(one_hot(true_classes[anchor], 8))
            other_classes.append(one_hot(true_classes[other], 8))


    anchors = torch.stack(anchors).float()
    others = torch.stack(others).float()
    similarity = torch.FloatTensor(similarity)
    anchor_classes = torch.stack(anchor_classes)
    other_classes = torch.stack(other_classes)

    input = [anchors, others]
    gt = [similarity, anchor_classes, other_classes]
    return input, gt

def build_channels(img, stats, center, disk_area, output_size):
    patches = []
    center = center[::-1]
    area = stats[-1]
    lat, lon = px_to_latlon(center, img.shape[0] // 2)
    patches.append(centered_patch(img, center, output_size))
    patches.append(np.full(output_size, area/disk_area))
    patches.append(np.full(output_size, lat))
    patches.append(np.full(output_size, lon))
    return torch.stack([torch.FloatTensor(p) for p in patches])

def centered_patch(img, c, output_size):
    w, h = [s//2 for s in output_size]
    padded = np.pad(img, ((w,w),(h,h)), mode='constant')
    c = [int(c[0]+w), int(c[1]+h)]
    return padded[c[0]-w:c[0]+w,c[1]-h:c[1]+h]

def px_to_latlon(c, radius):
    lat = math.acos((c[0] - radius) / radius)
    lon = math.acos((c[1] - radius) / radius)
    return (lat, lon)

def one_hot(idx, num_classes):
    oh = (idx == torch.arange(num_classes).reshape(1, num_classes)).float()
    return oh


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
