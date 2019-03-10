import cv2
from utils.data_vis import plot_mask
import os

mask_dir = 'homeRAID/efini/dataset/SDO/masks'
image_dir = 'homeRAID/efini/dataset/SDO/products'

for file in os.listdir(image_dir)[:2]:
    masks = [m.split('_mask.')[0] for m in os.listdir(mask_dir)]
    if file.split('.')[0] in masks:
        img = cv2.imread(os.path.join(image_dir, file), 0)
        mask = cv2.imread(os.path.join(mask_dir, file.split('.')[0]+'_mask.png'), 0)
        plot_mask(img, mask//200).show()
