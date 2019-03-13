import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# FENNEC'S VISUALIZATION functions

def plot_mask(img, mask):
    im = img.copy()
    mask = np.array(mask * 255, dtype=np.uint8, copy=True)
    mask = np.dstack((mask,mask,mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    im2,contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(im, contour, -1, (255,0,0), thickness = 1)
    img3 = np.dstack((im,im,im)).copy()
    b,g,r = cv2.split(img3)
    r = cv2.add(b, 30, dst = b, mask = binary, dtype = cv2.CV_8U)
    cv2.merge((b,g,r), img3)
    return img3

# -----------------------------------------------------------------------------

def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()
