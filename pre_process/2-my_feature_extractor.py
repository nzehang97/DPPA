import os
import time

import h5py
import numpy as np
import openslide
import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi):
        self.wsi = wsi
        self.roi_transforms = trnsfrms_val
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords'][:]

            self.coords = dset
            self.length = len(dset)
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']

    def __len__(self):
        return self.length

    def summary(self):
        dset = self.coords
        for name, value in dset.attrs.items():
            print(name, value)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.roi_transforms(img)
        return img, coord


if __name__ == '__main__':
    device = 'cuda'
    model = ctranspath()
    model.head = nn.Identity()
    model = model.to(device)
    td = torch.load(r'./ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()
    batch_size = 128
    save_path = '../0_Extracted_feature'
    os.makedirs(save_path, exist_ok=True)

    slide_list = np.sort(os.listdir('../0_RESULTS_DIRECTORY/patches'))

    for i, s in enumerate(slide_list):
        slide_name = s.split('.')[0]
        print('\nprogress: {}/{}-----------------{}'.format(i, len(slide_list), slide_name))

        time1 = time.time()
        file_path = '../0_RESULTS_DIRECTORY/patches/' + s
        output_path = save_path + '/' + s

        slide_file_path = f'../Datasets/TCGA_GBMLGG/{slide_name}.svs'
        wsi = openslide.open_slide(slide_file_path)

        test_datat = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi)
        database_loader = torch.utils.data.DataLoader(test_datat, batch_size=batch_size, shuffle=False)

        mode = 'w'
        with torch.no_grad():
            for count, (batch, coords) in enumerate(database_loader):
                if count % 20 == 0:
                    print('batch {}/{}, {} files processed'.format(count, len(database_loader), count * batch_size))
                fea = model(batch.to(device))
                features = fea.cpu().numpy().squeeze()
                asset_dict = {'features': features, 'coords': coords.numpy()}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

        time_elapsed = time.time()-time1
        print(f'computing features for {slide_name} took {time_elapsed: .2f} s')
