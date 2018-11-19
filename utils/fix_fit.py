from astropy.io import fits
import os, sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

for i in os.listdir(in_dir):
    fn = os.path.join(in_dir, i)
    if '.fit' in fn:
        hdul = fits.open(fn, ignore_missing_end=True)
        hdul.writeto(os.path.join(out_dir, i), output_verify="fix+ignore")
