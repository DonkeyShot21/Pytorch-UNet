import cv2
from utils.data_vis import plot_mask
import os, time, sys

mask_dir = '/homeRAID/efini/dataset/{}/masks'.format(sys.argv[1])
image_dir = '/homeRAID/efini/dataset/{}/images'.format(sys.argv[1])

for file in os.listdir(image_dir)[::-1]:
    masks = [m.split('_mask.')[0] for m in os.listdir(mask_dir)]
    if file.split('.')[0] in masks:
        print(file)
        img = cv2.imread(os.path.join(image_dir, file), 0)
        mask = cv2.imread(os.path.join(mask_dir, file.split('.')[0]+'_mask.png'), 0)
        plot_mask(img, mask//200).show()
        input("Press Enter to continue...")
