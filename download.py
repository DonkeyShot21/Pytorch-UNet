#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from astropy.io import fits
import sunpy.map
from sunpy.time import parse_time

from astropy.units import Quantity
from sunpy.map import Map

import cv2, torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os, math, heapq, sys
import numpy as np
from torchvision.transforms import Compose

from utils.load import search_VSO, remove_if_exists
from utils.utils import slice, rotate_coord, keep_best, normalize_map, normalize_img, to_uint8
from utils.data_vis import plot_mask

import warnings
warnings.filterwarnings('ignore')




def create_dataset_SDO(SIDC_filename, fenyi_dir):

    sidc_csv = pd.read_csv(SIDC_filename, sep=';', header=None)
    sidc_csv.drop(sidc_csv[[3,5,6,7]], axis=1, inplace=True)
    sidc_csv.astype(np.int32)

    fenyi_sunspot = []
    years = []
    for fenyi_fn in os.listdir(fenyi_dir):
        years.append(int(fenyi_fn[4:8]))
        fn = os.path.join(fenyi_dir,fenyi_fn)
        fenyi_sunspot.append(pd.read_csv(fn, sep=','))
    fenyi_sunspot = pd.concat(fenyi_sunspot)

    sidc_csv = sidc_csv[sidc_csv[0].isin(years)]


    for index, row in sidc_csv.iterrows():
        create_image_SDO(row, fenyi_sunspot)


def create_image_SDO(row, fenyi_sunspot):
    row = row.to_frame().transpose()

    # sampling with probability from SIDC
    print("Sampling from SIDC...")
    #row = sidc_csv.sample(weights=sidc_csv[4])
    day = '/'.join(map(str, row.iloc[0][:-1]))
    date = datetime.strptime(day + ' 12:00:00', '%Y/%m/%d %H:%M:%S')

    # loading sunspot data from DPD
    print("Loading sunspot data...", date)
    dpd = fenyi_sunspot.query(("year == @date.year & "
                               "month == @date.month & "
                               "day == @date.day"))
    dpd = dpd[(dpd[['projected_umbra',
                    'projected_whole_spot']].T != 0).any()]

    ws = dpd[['projected_whole_spot', 'group_number', 'group_spot_number']]
    rp = dpd[['position_angle', 'center_distance']]

    for index, row in ws.iterrows():
        wsa = row['projected_whole_spot']
        if wsa < 0:
            match = ws.query(("group_number == @row.group_number & "
                              "group_spot_number == -@wsa"))
            area = match['projected_whole_spot'].iloc[0]
            ws.loc[row.name,'projected_whole_spot'] = area

    groups = list(ws['group_number'].unique())
    time = datetime.strptime('-'.join([str(i) for i in list(dpd.iloc[0])[1:7]]), '%Y-%m-%d-%H-%M-%S')

    # SDO

    dir = 'homeRAID/efini/dataset/SDO/images'
    dir_out = 'homeRAID/efini/dataset/SDO/products'
    dir_mask_out = 'homeRAID/efini/dataset/SDO/masks'


    start_time = (time - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
    end_time = (time + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')

    try:
        print("Searching VSO...")
        file = search_VSO(start_time, end_time, dir)
        print(file)
        hmi = Map(file)
    except Exception as e:
        print(e)
        return

    # get the data from the maps
    img = normalize_map(hmi)

    # get the coordinates and the date of the sunspots from DPD
    print("Creating mask...")
    ss_coord = dpd[['heliographic_latitude', 'heliographic_longitude']]
    ss_date = parse_time(time)
    sunspots = rotate_coord(hmi, ss_coord, ss_date)

    # mask = (255 * img_cont).astype(np.uint8)
    instances = np.zeros(img.shape, dtype=np.float32)
    mask = np.zeros(img.shape, dtype=np.float32)

    disk_mask = np.where(255*img > 10)
    disk_mask = {(c[0],c[1]) for c in np.column_stack(disk_mask)}
    disk_mask_num_px = len(disk_mask)

    for i in range(len(sunspots)):
        print(i, len(sunspots))
        o = 4 # offset
        p = sunspots[i]

        group_idx = groups.index(ws.iloc[i]['group_number'])
        patch = img[int(p[1])-o:int(p[1])+o,int(p[0])-o:int(p[0])+o]
        low = np.where(patch == np.amin(patch))

        center = (img.shape[0] / 2, img.shape[1] / 2)
        distance = np.linalg.norm(tuple(j-k for j,k in zip(center,p)))
        cosine_amplifier = math.cos(math.radians(1) * distance / center[0])
        norm_num_px = cosine_amplifier * ws.iloc[i]['projected_whole_spot']
        ss_num_px = 8.6 * norm_num_px * disk_mask_num_px / 10e6

        new = set([(p[1] - o + low[1][0], p[0] - o + low[0][0])])
        whole_spot = set()
        candidates = set()
        expansion_rate = 3
        while len(whole_spot) < ss_num_px:
            expand = {(n[0]+i,n[1]+j)
                      for i in [-1,0,1]
                      for j in [-1,0,1]
                      for n in new}
            for e in set(expand - whole_spot):
                candidates.add(e)
            new = heapq.nsmallest(expansion_rate, candidates, key=lambda k: img[k])
            for n in new:
                candidates.remove(n)
            whole_spot.update(set(new))

        for c in set.intersection(whole_spot, disk_mask):
            instances[c] = 255 - group_idx


    x, y = [c[0] for c in disk_mask], [c[1] for c in disk_mask]

    minx, maxx, miny, maxy = np.amin(x), np.amax(x), np.amin(y), np.amax(y)

    print(minx, maxx, miny, maxy)

    top_left, bottom_right = (minx, miny), (maxx, maxy)

    img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    img = cv2.resize(img, (4000, 4000))

    instances = instances[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    instances = cv2.resize(instances, (4000, 4000))

    out_filename = file.split('.')[0].split('\\')[-1]

    cv2.imwrite(os.path.join(dir_out,out_filename+'.png'), ((img*2**16) -1).astype(np.uint16))
    Image.fromarray(instances.astype(np.uint8)).save(os.path.join(dir_mask_out,out_filename+'_mask.png'))






def create_dataset_ground(SIDC_filename, fenyi_dir):
    #GROUND
    sidc_csv = pd.read_csv(SIDC_filename, sep=';', header=None)
    sidc_csv.drop(sidc_csv[[3,5,6,7]], axis=1, inplace=True)
    sidc_csv.astype(np.int32)

    fenyi_sunspot = []
    years = []
    for fenyi_fn in os.listdir(fenyi_dir):
        years.append(int(fenyi_fn[4:8]))
        fn = os.path.join(fenyi_dir,fenyi_fn)
        fenyi_sunspot.append(pd.read_csv(fn, sep=','))
    fenyi_sunspot = pd.concat(fenyi_sunspot)

    sidc_csv = sidc_csv[sidc_csv[0].isin(years)]


    for index, row in sidc_csv.iterrows():
        create_image_ground(row, fenyi_sunspot)





def create_image_ground(row, fenyi_sunspot):

    row = row.to_frame().transpose()

    # sampling with probability from SIDC
    print("Sampling from SIDC...")
    #row = sidc_csv.sample(weights=sidc_csv[4])
    day = '/'.join(map(str, row.iloc[0][:-1]))
    date = datetime.strptime(day + ' 12:00:00', '%Y/%m/%d %H:%M:%S')

    # loading sunspot data from DPD
    print("Loading sunspot data...", date)
    dpd = fenyi_sunspot.query(("year == @date.year & "
                               "month == @date.month & "
                               "day == @date.day"))
    dpd = dpd[(dpd[['projected_umbra',
                    'projected_whole_spot']].T != 0).any()]

    ws = dpd[['projected_whole_spot', 'group_number', 'group_spot_number']]
    rp = dpd[['position_angle', 'center_distance']]

    for index, row in ws.iterrows():
        wsa = row['projected_whole_spot']
        if wsa < 0:
            match = ws.query(("group_number == @row.group_number & "
                              "group_spot_number == -@wsa"))
            area = match['projected_whole_spot'].iloc[0]
            ws.loc[row.name,'projected_whole_spot'] = area

    groups = list(ws['group_number'].unique())
    time = datetime.strptime('-'.join([str(i) for i in list(dpd.iloc[0])[1:7]]), '%Y-%m-%d-%H-%M-%S')

    dir = '/homeRAID/efini/dataset/ground/images'
    dir_out = '/homeRAID/efini/dataset/ground/products'
    dir_mask_out = '/homeRAID/efini/dataset/ground/masks'

    files = os.listdir(dir)
    file = [os.path.join(dir,f) for f in files if time.strftime('%Y%m%d') in f][0]
    if file.split('/')[-1].split('.')[0]+'.png' in os.listdir(dir_out):
        return
    print(file)
    with fits.open(file, ignore_missing_end=True) as hdul:
        # hdul.info()
        img = np.flip(-hdul[0].data.astype(np.float64),0)
        center = (int(hdul[0].header['CENT_C']), int(hdul[0].header['CENT_R']))
        radius = int(hdul[0].header['R_SUN'])
        if hdul[0].header['TELESCOP'] == 'Kanzelhoehe':
            tilt = -hdul[0].header['PS']
        else:
            tilt = -hdul[0].header['P']

    min = np.amin(img)
    range = np.amax(img) - min

    disk = cv2.circle(np.zeros(img.shape),center,radius,1,-1)
    disk_mask = {(c[0],c[1]) for c in np.column_stack(np.where(disk>0))}
    disk_mask_num_px = len(disk_mask)
    img = img - min
    img[disk==0] = 0
    img = normalize_img(img)
    rot_mat = cv2.getRotationMatrix2D(center,tilt,1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[::-1])

    instances = np.zeros(img.shape)

    print('Creating mask...')
    for i, row in rp.iterrows():
        pa = math.radians(row['position_angle'])
        r = row['center_distance']
        coord = (int(center[0]-radius*r*math.sin(pa)),
                 int(center[1]-radius*r*math.cos(pa)))

        o = 4 # offset
        p = coord

        group_idx = groups.index(ws.loc[i]['group_number'])
        patch = img[p[0]-o:p[0]+o,p[1]-o:p[1]+o]
        low = np.where(patch == np.amin(patch))

        distance = np.linalg.norm(tuple(j-k for j,k in zip(center,p)))
        cosine_amplifier = math.cos(math.radians(1) * distance / center[0])
        norm_num_px = cosine_amplifier * ws.loc[i]['projected_whole_spot']
        ss_num_px = 8.8 * norm_num_px * disk_mask_num_px / 10e6

        new = set([(p[1] - o + low[1][0], p[0] - o + low[0][0])])
        whole_spot = set()
        candidates = set()
        expansion_rate = 3
        while len(whole_spot) < ss_num_px:
            expand = {(n[0]+i,n[1]+j)
                      for i in [-1,0,1]
                      for j in [-1,0,1]
                      for n in new}
            for e in set(expand - whole_spot):
                candidates.add(e)
            new = heapq.nsmallest(expansion_rate, candidates, key=lambda k: img[k])
            for n in new:
                candidates.remove(n)
            whole_spot.update(set(new))

        for c in set.intersection(whole_spot, disk_mask):
            instances[c] = 255 - group_idx


    half_img = int(radius)

    top_left = (center[1]-half_img, center[0]-half_img)
    bottom_right =  (center[1]+half_img, center[0]+half_img)

    img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    img = cv2.resize(img, (4000, 4000))

    instances = instances[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    instances = cv2.resize(instances, (4000, 4000))

    out_filename = file.split('.')[0].split('\\')[-1]

    cv2.imwrite(os.path.join(dir_out,out_filename+'.png'), ((img*2**16) -1).astype(np.uint16))
    Image.fromarray(instances.astype(np.uint8)).save(os.path.join(dir_mask_out,out_filename+'_mask.png'))




if __name__ == '__main__':

    if sys.argv[1] == 'SDO':
        create_dataset_SDO('data/sidc/SIDC_dataset.csv',
                           'data/dpd/')
    if sys.argv[1] == 'ground':
        create_dataset_ground('data/sidc/SIDC_dataset.csv',
                             'data/dpd/')







## QUERYING AND DRAWING HEK


# fig = plt.figure()
# ax = plt.subplot(projection=hmi)
# hmi.plot(axes=ax)
#
# for ss in responses:
#     p = [v.split(" ") for v in ss["hpc_boundcc"][9:-2].split(',')]
#     ss_date = parse_time(ss['event_starttime'])
#
#     ss_boundary = SkyCoord(
#         [(float(v[0]), float(v[1])) * u.arcsec for v in p],
#         obstime=ss_date,
#         frame=frames.Helioprojective)
#     rotated_ss_boundary = solar_rotate_coordinate(ss_boundary, hmi.date)
#
#     px = hmi.world_to_pixel(rotated_ss_boundary)
#     points = [[(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]]
#     cv2.fillPoly(img, np.array(points), 127)
#
#
#     # ax.fill(hmi.world_to_pixel(rotated_ss_boundary),'b')
#     ax.plot_coord(rotated_ss_boundary, color='c')
#
# ax.set_title('{:s}\n{:s}'.format(hmi.name, ss['frm_specificid']))
# plt.colorbar()
# plt.show()


# 0.2650327216254339 3.795
