import os
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# 加载中心数据
center_datas = loadmat('kmeans_256.mat')
center_datas = center_datas['features']

# 获取所有slide路径
slides = os.listdir('../0_Cluster_data/features_100')
slide_paths = [os.path.join('../0_Extracted_feature', s) for s in slides]
os.makedirs('patch_class_index', exist_ok=True)

# 定义处理每个slide的函数
def process_slide(slide_path):

    slide = os.path.basename(slide_path)
    print(slide)
    if os.path.exists(f'patch_class_index/{slide}'):
        return
    # 特征提取和分类
    with h5py.File(slide_path, 'r') as f:
        features = f['features'][:]
    patch_class_index = []
    for feature in features:
        cos_sims = cosine_similarity(feature.reshape(1, -1), center_datas)
        idx = np.argmax(cos_sims)
        patch_class_index.append(idx)

    # 保存结果
    with h5py.File(f'patch_class_index/{slide}', 'w') as ff:
        ff['index'] = np.array(patch_class_index)


# 使用joblib并行处理slide
from joblib import Parallel, delayed
num_cores = 10

Parallel(n_jobs=num_cores)(delayed(process_slide)(path) for path in slide_paths)
