import os
import h5py
import time
import openslide
import numpy as np
from sklearn.cluster import KMeans


def get_feature(CLASS_NUM, fea_path, var_index):
    if os.path.isfile(fea_path):
        with h5py.File(fea_path, 'r') as hdf5_file:
            feature = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        with h5py.File(f'../0_RESULTS_DIRECTORY/patches/{os.path.basename(fea_path)}', 'r') as hdf5_file:
            coords_res = hdf5_file['coords'][:]

        if len(coords_res) < CLASS_NUM:
            padding_rows = CLASS_NUM - len(coords_res)
            num_rows_to_sample = 5

            new_rows = []
            while len(new_rows) < padding_rows:
                random_indices = np.random.choice(len(coords_res), num_rows_to_sample, replace=False)
                sampled_rows = feature[random_indices]
                mean_row = np.mean(sampled_rows, axis=0)
                new_rows.append(mean_row)

            new_rows = np.array(new_rows)
            new_feature = np.concatenate((feature, new_rows), axis=0)
            return new_feature, coords

        return feature, coords


def chuster_concat(feas_path, CLASS_NUM, slide, var_index):
    print('\ncluster--', slide)
    fea_path = os.path.join(feas_path, slide + '.h5')
    feature_raw, coords_raw = get_feature(CLASS_NUM, fea_path, var_index)

    print(f'Kmeans--{CLASS_NUM}类--start')
    start_time = time.time()
    kmeans = KMeans(n_clusters=CLASS_NUM, random_state=0).fit(feature_raw)
    print(f'spend-{time.time() - start_time}')
    print(f'Kmeans--{CLASS_NUM}类--end')
    label = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    with h5py.File(f'{cluster_path}/kmeans_{CLASS_NUM}_index/{slide_dir}.h5', 'w') as hh:
        hh[f'{slide_dir}'] = label
    feature_49 = []
    coord_49 = []
    for j in range(CLASS_NUM):
        clu_cen_j = cluster_centers[j, :].reshape(-1, 1)
        class_j_idx = np.array([idx for idx, l in enumerate(label) if l == j])

        temp_path_perm = np.tile(clu_cen_j, len(class_j_idx))
        sam_j = feature_raw[class_j_idx, :].T
        sum_data = np.sum((sam_j - temp_path_perm) ** 2, axis=0)
        a = np.argmin(sum_data)

        feature_49.append(feature_raw[class_j_idx[a]])
        coord_49.append(coords_raw[class_j_idx[a]])
    assert len(feature_49) == CLASS_NUM
    with h5py.File(f'{cluster_path}/features_{CLASS_NUM}/{slide}.h5', 'w') as ff:
        ff[f'features'] = feature_49
        ff[f'coords'] = coord_49


if __name__ == '__main__':
    CLASS_NUM = 100
    print(os.getcwd())
    feas_path = '../0_Extracted_feature'
    cluster_path = f'../0_Cluster_data'

    total_Slide = len(os.listdir(feas_path))
    count = 0
    os.makedirs(cluster_path, exist_ok=True)
    os.makedirs(f'{cluster_path}/features_{CLASS_NUM}', exist_ok=True)
    os.makedirs(f'{cluster_path}/kmeans_{CLASS_NUM}_index', exist_ok=True)

    for file in np.sort(os.listdir(feas_path)):
        patch_h5 = os.path.join(feas_path, file)
        fea_h5 = file.split('.')[0]+'.h5'
        if os.path.exists(f'{cluster_path}/{fea_h5}'):
            count += 1
            continue
        slide_dir = file.split('.')[0]

        count += 1
        print('\n-----------------------------------')
        print(f'{slide_dir}  {count}/{total_Slide}')

        chuster_concat(feas_path, CLASS_NUM, slide_dir, 0)

