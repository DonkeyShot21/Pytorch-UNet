from __future__ import print_function, division
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sunpy.map
from sunpy.time import parse_time

from astropy.units import Quantity
from astropy.io import fits
from sunpy.map import Map

import cv2, torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os, math, heapq
import numpy as np
from torchvision.transforms import Compose

from utils.load import search_VSO, normalize_map
from utils.load import remove_if_exists, inverse_normalize_map
from utils.utils import slice, rotate_coord, keep_best

import warnings
warnings.filterwarnings('ignore')




class HelioDataset(Dataset):
    def __init__(self, SIDC_filename, fenyi_dir, n_samples):
        super(Dataset, self).__init__()
        self.n_samples = n_samples

        sidc_csv = pd.read_csv(SIDC_filename, sep=';', header=None)
        sidc_csv.drop(sidc_csv[[3,5,6,7]], axis=1, inplace=True)
        sidc_csv.astype(np.int32)

        self.fenyi_sunspot = []
        years = []
        for fenyi_fn in os.listdir(fenyi_dir):
            years.append(int(fenyi_fn[4:8]))
            fn = os.path.join(fenyi_dir,fenyi_fn)
            self.fenyi_sunspot.append(pd.read_csv(fn, sep=','))
        self.fenyi_sunspot = pd.concat(self.fenyi_sunspot)

        self.sidc_csv = sidc_csv[sidc_csv[0].isin(years)]


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # sampling with probability from SIDC
        print("Sampling from SIDC...")
        row = self.sidc_csv.sample(weights=self.sidc_csv[4])
        day = '/'.join(map(str, row.iloc[0][:-1]))
        date = datetime.strptime(day + ' 12:00:00', '%Y/%m/%d %H:%M:%S')

        # loading sunspot data from DPD
        print("Loading sunspot data...", date)
        dpd = self.fenyi_sunspot.query(("year == @date.year & "
                                        "month == @date.month & "
                                        "day == @date.day"))
        dpd = dpd[(dpd[['projected_umbra','projected_whole_spot']].T != 0).any()]

        time = '-'.join([str(i) for i in list(dpd.iloc[0])[1:7]])
        time = datetime.strptime(time, '%Y-%m-%d-%H-%M-%S')

        if time > datetime(2010,2,11,0,0): # SDO launch date
            try:
                print("Searching VSO...")
                start_time = (time - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
                end_time = (time + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
                continuum_file, magnetic_file = None, None
                continuum_file, magnetic_file = search_VSO(start_time, end_time)
                hmi_cont = Map(continuum_file)
                hmi_mag = Map(magnetic_file)
            except Exception as e:
                print(e)
                remove_if_exists(continuum_file)
                remove_if_exists(magnetic_file)
                return self.__getitem__(idx)
        else: #otherwise use SOHO
            img_fns = {}
            for kind in ['magnetic', 'intensity']:
                dir = os.path.join('data/dpd/images', kind)
                fns = os.listdir(dir)
                img_fn = [i for i in fns if time.strftime('%Y%m%d') in i][0]
                img_fn = os.path.join(dir, img_fn)
                img_fns[kind] = img_fn
            hmi_cont = Map(img_fns['intensity'])
            hmi_mag = Map(img_fns['magnetic'])

        # get the data from the maps
        img_cont = inverse_normalize_map(hmi_cont.data)
        img_mag = normalize_map(hmi_mag.data)
        inputs = [img_cont, img_mag]

        # get the coordinates and the date of the sunspots from DPD
        print("Creating mask...")
        ss_coord = dpd[['heliographic_latitude', 'heliographic_longitude']]
        ss_date = parse_time(time)
        sunspots = rotate_coord(hmi_cont, ss_coord, ss_date)

        # mask = (255 * img_cont).astype(np.uint8)
        mask = np.zeros(img_cont.shape, dtype=np.float32)

        ws = dpd[['projected_whole_spot',
                  'group_number',
                  'group_spot_number']]

        for index, row in ws.iterrows():
            wsa = row['projected_whole_spot']
            if wsa < 0:
                match = ws.query(("group_number == @row.group_number & "
                                  "group_spot_number == -@wsa"))
                area = match['projected_whole_spot'].iloc[0]
                ws.loc[row.name,'projected_whole_spot'] = area

        groups = list(ws['group_number'].unique())
        disk_mask = np.where(255*img_cont > 10)
        disk_mask = {(c[0],c[1]) for c in np.column_stack(disk_mask)}
        disk_mask_num_px = len(disk_mask)
        whole_spot_mask = set()

        print(sunspots)

        for i in range(len(sunspots)):
            o = 4 # offset
            p = sunspots[i]
            # g_number = groups.index(ws.iloc[i]['group_number'])
            group = img_cont[int(p[1])-o:int(p[1])+o,int(p[0])-o:int(p[0])+o]
            low = np.where(group == np.amin(group))

            center = (img_cont.shape[0] / 2, img_cont.shape[1] / 2)
            distance = np.linalg.norm(tuple(j-k for j,k in zip(center,p)))
            cosine_amplifier = math.cos(math.radians(1) * distance / center[0])
            norm_num_px = cosine_amplifier * ws.iloc[i]['projected_whole_spot']
            ss_num_px = 8.6 * norm_num_px * disk_mask_num_px / 10e6

            new = set([(p[1] - o + low[1][0], p[0] - o + low[0][0])])
            whole_spot = set()
            candidates = dict()
            expansion_rate = 3
            while len(whole_spot) < ss_num_px:
                expand = {(n[0]+i,n[1]+j)
                          for i in [-1,0,1]
                          for j in [-1,0,1]
                          for n in new}
                for e in set(expand - whole_spot):
                    candidates[e] = img_cont[e]
                new = heapq.nsmallest(expansion_rate, candidates.keys(), key=lambda k: img_cont[k])
                #new = sorted(candidates, key=candidates.get)[:expansion_rate]
                for n in new:
                    candidates.pop(n, None)
                whole_spot.update(set(new))

            whole_spot_mask.update(whole_spot)

        for c in set.intersection(whole_spot_mask, disk_mask):
            mask[c] = 1

        mask = cv2.resize(mask, (4096,4096), interpolation=cv2.INTER_NEAREST)
        for i in range(len(inputs)):
            inputs[i] = cv2.resize(inputs[i], (4096,4096),
                                   interpolation=cv2.INTER_AREA)
            print(np.amax(inputs[i]), np.amin(inputs[i]))
            Image.fromarray(255*inputs[i]).show()
        inputs = np.array(inputs, dtype=np.float32)

            # mag_patch = cv2.resize(mag_patch, (1024,1024),
            #                 interpolation=cv2.INTER_AREA)
            # mask_patch = cv2.resize(mask_patch, (1024,1024),
            #                 interpolation=cv2.INTER_NEAREST)

        # show_mask(img_cont, mask)
        # remove_if_exists(continuum_file)
        # remove_if_exists(magnetic_file)

        return {"img": inputs.astype(np.float32), "mask": mask}



if __name__ == '__main__':

    dataset = HelioDataset('data/SIDC_dataset.csv',
                           'data/fenyi',
                           10)

    data_loader = DataLoader(dataset)

    for idx, batch_data in enumerate(data_loader):
        print(idx)
        print(batch_data['img'].size())
        print(batch_data['mask'].size())







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
