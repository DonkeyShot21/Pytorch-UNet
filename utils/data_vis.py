import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# FENNEC'S VISUALIZATION functions

def plot_mask(img, mask):
    mask = np.array(mask * 255, dtype=np.uint8)
    mask = np.dstack((mask,mask,mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    im2,contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(img, contour, -1, (255,0,0), thickness = 1)
    img = np.dstack((img,img,img))
    b,g,r = cv2.split(img)
    r = cv2.add(b, 30, dst = b, mask = binary, dtype = cv2.CV_8U)
    cv2.merge((b,g,r), img)
    return Image.fromarray(img)

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
