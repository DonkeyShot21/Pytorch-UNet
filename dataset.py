import cv2, os
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from utils.utils import patchify, sample_patches, sample_sunspot_pairs

class HelioDataset(Dataset):
    def __init__(self, sidc_path, ground_dir, SDO_dir, patch_size=200,
                 overlap=0.5, sunspots_per_patch=10):
        super(Dataset, self).__init__()

        self.overlap = overlap
        self.sunspots_per_patch = sunspots_per_patch
        self.patch_size = patch_size

        self.sidc = pd.read_csv(sidc_path, sep=';', header=None)
        self.sidc.drop(self.sidc[[3,5,6,7]], axis=1, inplace=True)
        self.sidc = self.sidc.astype(np.int32)
        dates = pd.to_datetime(self.sidc[0]*10000+self.sidc[1]*100+self.sidc[2],
                               format='%Y%m%d')
        self.sidc['date'] = dates
        self.sidc['sunspot_number'] = self.sidc[4]
        self.sidc.drop(self.sidc[[0,1,2,4]], axis=1, inplace=True)
        self.sidc.set_index('date', inplace=True, drop=True)

        self.data = []

        # explore ground directory, match masks and extract sunspot number
        for img_fn in os.listdir(os.path.join(ground_dir,'images')):
            img_path = os.path.join(ground_dir, 'images', img_fn)
            date = datetime.strptime(img_fn.split('_')[0],'%Y%m%d')
            sunspot_number = self.sidc.loc[date]['sunspot_number']
            if sunspot_number < sunspots_per_patch: continue
            mask_fn = img_fn.replace('.png','_mask.png')
            mask_path = os.path.join(ground_dir, 'masks', mask_fn)
            self.data.append({'img_path': img_path,
                              'sunspot_number': sunspot_number,
                              'mask_path': mask_path,
                              'date': date.strftime('%Y-%m-%d')})

        # explore SDO directory, match masks and extract sunspot number
        for img_fn in os.listdir(os.path.join(SDO_dir,'images')):
            img_path = os.path.join(SDO_dir, 'images', img_fn)
            date = datetime.strptime(img_fn.split('_')[1],'%Y%m%d')
            sunspot_number = self.sidc.loc[date]['sunspot_number']
            if sunspot_number < sunspots_per_patch: continue
            mask_fn = img_fn.replace('.png','_mask.png')
            mask_path = os.path.join(SDO_dir, 'masks', mask_fn)
            self.data.append({'img_path': img_path,
                              'sunspot_number': sunspot_number,
                              'mask_path': mask_path,
                              'date': date.strftime('%Y-%m-%d')})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            full_disk = cv2.imread(self.data[idx]['img_path'], -1)
            full_disk = full_disk.astype(np.float32)/ np.amax(full_disk)
            ground_truth = cv2.imread(self.data[idx]['mask_path'], -1)
            full_disk_mask = np.clip(ground_truth[:,:,2], 0, 1)
            #
            # sunspot_number = self.data[idx]['sunspot_number']
            # num_patches = sunspot_number // self.sunspots_per_patch
            # patches, masks = patchify(full_disk, full_disk_mask,
            #                           patch_size=self.patch_size,
            #                           overlap=self.overlap)
            # patches, masks = sample_patches(patches, masks, num_patches)
            # siamese_input, siamese_gt = sample_sunspot_pairs(full_disk,
            #                                                  full_disk_mask,
            #                                                  ground_truth[:,:,2],
            #                                                  ground_truth[:,:,1],
            #                                                  num_anchors=1)
            return {'full_disk': full_disk,
                    'full_disk_mask': full_disk_mask,
                    'full_disk_instances': ground_truth[:,:,2],
                    'full_disk_classes': ground_truth[:,:,1],
                    # 'patches': patches,
                    # 'masks': masks,
                    # 'anchors': siamese_input[0],
                    # 'others': siamese_input[1],
                    # 'class_others': siamese_gt[2],
                    # 'similarity': siamese_gt[0],
                    # 'sunspot_number': sunspot_number,
                    'date': self.data[idx]['date']}

        except Exception as e:
            print(e)
            print('Error on image', self.data[idx]['img_path'])
            with open('/homeRAID/efini/logs/corrupted.txt', 'a+') as f:
                f.write(self.data[idx]['img_path'] + ' ' + self.data[idx]['mask_path'] + '\n')
            return 0


class HelioDatasetVal(Dataset):
    def __init__(self, sidc_path, ground_dir, SDO_dir, patch_size=200,
                 overlap=0.5, sunspots_per_patch=10):
        super(Dataset, self).__init__()

        self.overlap = overlap
        self.sunspots_per_patch = sunspots_per_patch
        self.patch_size = patch_size

        self.sidc = pd.read_csv(sidc_path, sep=';', header=None)
        self.sidc.drop(self.sidc[[3,5,6,7]], axis=1, inplace=True)
        self.sidc = self.sidc.astype(np.int32)
        dates = pd.to_datetime(self.sidc[0]*10000+self.sidc[1]*100+self.sidc[2],
                               format='%Y%m%d')
        self.sidc['date'] = dates
        self.sidc['sunspot_number'] = self.sidc[4]
        self.sidc.drop(self.sidc[[0,1,2,4]], axis=1, inplace=True)
        self.sidc.set_index('date', inplace=True, drop=True)

        self.data = []

        # explore ground directory, match masks and extract sunspot number
        for img_fn in os.listdir(os.path.join(ground_dir,'images')):
            img_path = os.path.join(ground_dir, 'images', img_fn)
            date = datetime.strptime(img_fn.split('_')[0],'%Y%m%d')
            sunspot_number = self.sidc.loc[date]['sunspot_number']
            if sunspot_number < sunspots_per_patch: continue
            mask_fn = img_fn.replace('.png','_mask.png')
            mask_path = os.path.join(ground_dir, 'masks', mask_fn)
            self.data.append({'img_path': img_path,
                              'sunspot_number': sunspot_number,
                              'mask_path': mask_path,
                              'date': date.strftime('%Y-%m-%d')})

        # explore SDO directory, match masks and extract sunspot number
        for img_fn in os.listdir(os.path.join(SDO_dir,'images')):
            img_path = os.path.join(SDO_dir, 'images', img_fn)
            date = datetime.strptime(img_fn.split('_')[1],'%Y%m%d')
            sunspot_number = self.sidc.loc[date]['sunspot_number']
            if sunspot_number < sunspots_per_patch: continue
            mask_fn = img_fn.replace('.png','_mask.png')
            mask_path = os.path.join(SDO_dir, 'masks', mask_fn)
            self.data.append({'img_path': img_path,
                              'sunspot_number': sunspot_number,
                              'mask_path': mask_path,
                              'date': date.strftime('%Y-%m-%d')})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            full_disk = cv2.imread(self.data[idx]['img_path'], -1)
            full_disk = full_disk.astype(np.float32)/ np.amax(full_disk)
            ground_truth = cv2.imread(self.data[idx]['mask_path'], -1)
            full_disk_mask = np.clip(ground_truth[:,:,2], 0, 1)


            sunspot_number = self.data[idx]['sunspot_number']
            num_patches = sunspot_number // self.sunspots_per_patch
            patches, masks = patchify(full_disk, full_disk_mask,
                                      patch_size=self.patch_size,
                                      overlap=self.overlap)
            patches, masks = sample_patches(patches, masks, num_patches)
            siamese_input, siamese_gt = sample_sunspot_pairs(full_disk,
                                                             full_disk_mask,
                                                             ground_truth[:,:,2],
                                                             ground_truth[:,:,1],
                                                             num_anchors=1)
            return {'full_disk': full_disk,
                    'full_disk_mask': full_disk_mask,
                    'full_disk_instances': ground_truth[:,:,2],
                    'full_disk_classes': ground_truth[:,:,1],
                    'patches': patches,
                    'masks': masks,
                    'anchors': siamese_input[0],
                    'others': siamese_input[1],
                    'class_others': siamese_gt[2],
                    'similarity': siamese_gt[0],
                    'sunspot_number': sunspot_number,
                    'date': self.data[idx]['date']}

        except:
            print('Error on image', self.data[idx]['img_path'])
            with open('/homeRAID/efini/logs/corrupted.txt', 'a+') as f:
                f.write(self.data[idx]['img_path'] + ' ' + self.data[idx]['mask_path'] + '\n')
            return 0



if __name__ == '__main__':

    dataset = HelioDataset('data/sidc/SIDC_dataset.csv',
                           '/homeRAID/efini/dataset/ground',
                           '/homeRAID/efini/dataset/SDO')
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True)

    for idx, obs in enumerate(dataloader):
        print(idx)
        print(obs['patches'].size())
        print(obs['masks'].size())
        print(obs['full_disk'].size())
        print(obs['full_disk_instances'].size())
        print(obs['sunspot_number'])
        print(obs['date'])
