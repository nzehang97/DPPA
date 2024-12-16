import os
import h5py
import numpy as np
from tqdm import tqdm

K = 100
np.random.seed(2023)
phar = tqdm(os.listdir(f'../0_Cluster_data/features_100'))
os.makedirs(f'../0_Random_index', exist_ok=True)
for slide in phar:
    s = slide.split('.h5')[0]

    with h5py.File(f'../0_Cluster_data/kmeans_{K}_index/' + slide, 'r') as hdf5_file:
        kmeans_index = hdf5_file[s][:]

    index_list = np.array([])
    for epoch in range(100):
        index = np.array([])
        cluster_num = np.array([])
        for i in range(K):
            idx = np.where(kmeans_index == i)[0]
            cluster_num = np.append(cluster_num, len(idx))
            idx = np.random.choice(idx, 1, replace=True)
            index = np.append(index, idx)
        index = index.reshape(1, -1)
        index_list = index if len(index_list) == 0 else np.vstack((index_list, index))

    with h5py.File(f'../0_Random_index/{s}.h5', 'w') as f:
        f['index'] = index_list
