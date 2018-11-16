from __future__ import print_function, division
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sunpy.map
from sunpy.time import parse_time

from astropy.units import Quantity
from sunpy.map import Map

import cv2, torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os, math
import numpy as np
from torchvision.transforms import Compose

from utils.load import search_VSO, normalize_map, remove_if_exists
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

        time = datetime.strptime('-'.join([str(i) for i in list(dpd.iloc[0])[1:7]]), '%Y-%m-%d-%H-%M-%S')
        start_time = (time - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
        end_time = (time + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')

        try:
            print("Searching VSO...")
            continuum_file, magnetic_file = None, None
            continuum_file, magnetic_file = search_VSO(start_time, end_time)
            hmi_cont = Map(continuum_file)
            hmi_mag = Map(magnetic_file)
        except Exception as e:
            print(e)
            remove_if_exists(continuum_file)
            remove_if_exists(magnetic_file)
            return self.__getitem__(idx)

        # get the data from the maps
        img_cont = normalize_map(hmi_cont)
        img_mag = 2 * normalize_map(hmi_mag) - 1
        inputs = np.array([img_cont, img_mag])

        # get the coordinates and the date of the sunspots from DPD
        print("Creating mask...")
        ss_coord = dpd[['heliographic_latitude', 'heliographic_longitude']]
        ss_date = parse_time(time)
        sunspots = rotate_coord(hmi_cont, ss_coord, ss_date)

        # mask = (255 * img_cont).astype(np.uint8)
        mask = np.zeros(inputs.shape, dtype=np.float32)

        u_ws = dpd[['projected_umbra',
                  'projected_whole_spot',
                  'group_number',
                  'group_spot_number']].dropna()

        for index, row in u_ws.iterrows():
            for feature in ['projected_umbra', 'projected_whole_spot']:
                a = row[feature]
                if a < 0:
                    match = u_ws.query(("group_number == @row.group_number & "
                                        "group_spot_number == -@a"))
                    area = match[feature].iloc[0]
                    u_ws.loc[row.name,feature] = area

        groups = list(u_ws['group_number'].unique())
        disk_mask = np.where(255*img_cont > 10)
        disk_mask = {(c[0],c[1]) for c in np.column_stack(disk_mask)}
        disk_mask_num_px = len(disk_mask)
        whole_spot_mask = set()
        umbra_mask = set()

        for i in range(len(sunspots)):
            o = 4 # offset
            p = sunspots[i]
            # g_number = groups.index(ws.iloc[i]['group_number'])
            group = img_cont[int(p[1])-o:int(p[1])+o,int(p[0])-o:int(p[0])+o]
            low = np.where(group == np.amin(group))

            center = (img_cont.shape[0] / 2, img_cont.shape[1] / 2)
            distance = np.linalg.norm(tuple(j-k for j,k in zip(center,p)))
            cosine_amplifier = math.cos(math.radians(1) * distance / center[0])
            pws =  u_ws.iloc[i]['projected_whole_spot']
            norm_num_px = cosine_amplifier * pws
            ss_num_px = 8.6 * norm_num_px * disk_mask_num_px / 10e6
            umbra_num_px = ss_num_px * u_ws.iloc[i]['projected_umbra'] / pws

            new = set([(p[1] - o + low[1][0], p[0] - o + low[0][0])])
            whole_spot = set()
            umbra = set()
            candidates = dict()
            expansion_rate = 10
            while len(whole_spot) + len(umbra) < ss_num_px:
                expand = {(n[0]+i,n[1]+j)
                          for i in [-1,0,1]
                          for j in [-1,0,1]
                          for n in new}
                for e in set(expand - whole_spot -umbra):
                    candidates[e] = img_cont[e]
                new = sorted(candidates, key=candidates.get)[:expansion_rate]
                for n in new:
                    candidates.pop(n, None)
                    if len(umbra) >= umbra_num_px:
                        whole_spot.add(n)
                    else:
                        umbra.add(n)

            whole_spot_mask.update(whole_spot)
            umbra_mask.update(umbra)
            #umbra_candidates = {u:img_cont[u] for u in whole_spot}
            #umbra_candidates = sorted(umbra_candidates, key=umbra_candidates.get)[:int(umbra_num_px)]
            #umbra_mask.update(set(umbra_candidates))

        for c in set.intersection(whole_spot_mask, disk_mask):
            mask[0][c] = 1
        for c in set.intersection(umbra_mask, disk_mask):
            mask[1][c] = 1

        # show_mask(img_cont, mask)
        remove_if_exists(continuum_file)
        remove_if_exists(magnetic_file)

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
