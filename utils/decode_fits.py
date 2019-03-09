import numpy as np
import astropy.io.fits


min = np.amin(img)
img += min
max = np.amax(img)
background = img[0][0]


img[img==background] =
