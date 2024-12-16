import os
import h5py
import numpy as np
from scipy.io import savemat
from sklearn.cluster import KMeans
from tqdm import tqdm

fea_root = '../0_Cluster_data/features_100'
all_features = np.array([])

slide_list = os.listdir('../0_Cluster_data/features_100')
pbar = tqdm(np.sort(slide_list))
for slide in pbar:
    fea_path = f'{fea_root}/{slide}'
    with h5py.File(fea_path, 'r') as F:
        feature = F['features'][:]
    all_features = feature if len(all_features) == 0 else np.vstack((all_features, feature))

CLASS_NUM = 256
kmeans = KMeans(n_clusters=CLASS_NUM, random_state=0).fit(all_features)
label = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
feature_center = []
for j in range(CLASS_NUM):
    clu_cen_j = cluster_centers[j, :].reshape(-1, 1)
    class_j_idx = np.array([idx for idx, l in enumerate(label) if l == j])

    temp_path_perm = np.tile(clu_cen_j, len(class_j_idx))
    sam_j = all_features[class_j_idx, :].T
    sum_data = np.sum((sam_j - temp_path_perm) ** 2, axis=0)
    a = np.argmin(sum_data)

    feature_center.append(all_features[class_j_idx[a]])

feature_center = np.array(feature_center)
savemat('kmeans_256.mat', {'features': feature_center})
