import os
import numpy as np
import h5py
from scipy import stats
import pandas as pd
from utils.vis_utils.heatmap_utils import initialize_wsi, drawHeatmap
import cv2
import functools
import multiprocessing


def features_p_values(ff, task):
    if os.path.exists(f'{ff}/center_patch_p_value_{task}.csv'):
        return
    ground_truth = pd.read_csv(f'../model_train/data_label/label_{task}.csv')
    datas = list(ground_truth['slide_id'])
    labels = np.array(ground_truth['label'])

    class_index = 'patch_class_index'
    slides = os.listdir(class_index)

    all_ratio = np.array([])
    all_label = np.array([])
    for slide in slides:
        s = slide.split('.h5')[0]
        if s in datas:
            idx = datas.index(s)
        else:
            continue
        label = labels[idx]
        ratio = np.array([])
        with h5py.File(f'patch_class_index/{slide}', 'r') as f:
            a_index = f['index'][:]
        for i in range(256):
            a = np.sum(a_index == i) / len(a_index)
            ratio = np.append(ratio, a).reshape(1, -1)
        all_ratio = ratio if len(all_ratio) == 0 else np.vstack((all_ratio, ratio))
        all_label = np.append(all_label, label)

    df = pd.DataFrame(all_ratio, columns=[f'{i + 1:03d}' for i in range(256)])
    df['label'] = all_label

    p_values = []
    for col in df.columns[:-1]:
        aa = np.array(df[df['label'] == 0][col])
        bb = np.array(df[df['label'] == 1][col])
        aa = np.mean(aa)
        bb = np.mean(bb)
        rr = 0 if aa > bb else 1
        t, p = stats.ttest_ind(df[df['label'] == 0][col],
                               df[df['label'] == 1][col])
        p_values.append((col, p, t, rr, aa, bb))

    # Sort by p-value
    p_values.sort(key=lambda x: x[2])

    sss = [s[0] for s in p_values]
    p_value = [s[1] for s in p_values]
    t_statistic = [s[2] for s in p_values]
    ratio_max = [s[3] for s in p_values]
    aaa = [s[4] for s in p_values]
    bbb = [s[5] for s in p_values]
    sort = np.array(range(len(sss))) + 1

    pd.DataFrame({'sort': sort, 'class': sss, 'p_value': p_value, 't_statistic': t_statistic, 'ratio': ratio_max,
                  'aaa': aaa, 'bbb': bbb}).to_csv(f'{ff}/center_patch_p_value_{task}.csv')


def crop(slide, file, task):
    if os.path.exists(f'{file}/{task}_new_crop_RGB/{slide}'):
        return
    img = cv2.imread(f'{file}/{task}_RGB/{slide}')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.uint8(img_gray)
    img_mask = np.uint8(np.where(img_gray == 255, 0, 255))

    x, y, w, h = cv2.boundingRect(img_mask)

    img = img[y:y + h, x:x + w]
    img_gray = img_gray[y:y + h, x:x + w]
    img_mask = img_mask[y:y + h, x:x + w]

    if img_gray.shape == img_mask.shape:
        cv2.imwrite(f'{file}/{task}_new_crop_RGB/{slide}', img)
    else:
        print(slide, ' dim error')


def my_crop(file, task):
    os.makedirs(f'{file}/{task}_new_crop_RGB', exist_ok=True)
    imgs = os.listdir(f'{file}/{task}_RGB')
    pool = multiprocessing.Pool(10)
    for img in imgs:
        pool.apply_async(
            functools.partial(crop, img, file, task)
        )
    pool.close()
    pool.join()


def draw_heatmap(slide_id, ff, tumor):

    if os.path.exists(f'{ff}/{tumor}_RGB/{slide_id}.jpg'):
        return
    slide_path = f'../Datasets/TCGA_GBMLGG/{slide_id}.svs'

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': [],
                  'exclude_ids': []}
    filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 2}

    scale = 33
    try:
        wsi_object = initialize_wsi(slide_path, seg_params=seg_params, filter_params=filter_params, scale=scale)
    except:
        print(slide_id)
        return

    with h5py.File('../0_Extracted_feature/' + slide_id + '.h5', 'r') as hdf5_file:
        coords = hdf5_file['coords'][:]

    with h5py.File('patch_class_index/' + slide_id + '.h5', 'r') as hdf5_file:
        scores_buf = hdf5_file['index'][:]

    df = pd.read_csv(f'{ff}/center_patch_p_value_{tumor}.csv')
    no_index = {}
    for i in range(256):
        key = df['class'][i] - 1
        no_index[key] = df['sort'][i] - 1
    scores = []
    for s in scores_buf:
        scores.append(no_index[s])
    scores = np.array(scores)

    with h5py.File('../0_RESULTS_DIRECTORY/patches/' + slide_id + '.h5', 'r') as f:
        patch_size = f['coords'].attrs['patch_size']
    # 热图展示
    # cmap: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap='jet',
                          alpha=0.5, use_holes=True, binarize=False, vis_level=-1, scale=scale,
                          blank_canvas=True, thresh=-1, patch_size=patch_size, convert_to_percentiles=False)

    heatmap.save(f'{ff}/{tumor}_RGB/{slide_id}.jpg')


def my_heatmap(ff, task):
    np.random.seed(2023)
    ground_truth = pd.read_csv(f'../model_train/data_label/label_{task}.csv')
    datas = list(ground_truth['slide_id'])
    os.makedirs(f'map/{ff}/{task}_RGB', exist_ok=True)
    files = np.sort(os.listdir('../0_Cluster_data/features_100'))
    aa = [f.split('.h5')[0] for f in files if f.split('.h5')[0] in datas]

    pool = multiprocessing.Pool(10)
    for slide_id in aa:
        pool.apply_async(functools.partial(draw_heatmap, slide_id, ff, task))
    pool.close()
    pool.join()


if __name__ == '__main__':
    ff = 'sort_by_p_value'
    for task in ['IDH']:
        print('\nfeatures_p_values...')
        features_p_values(ff, task)
        print('\nheatmaping...')
        my_heatmap(ff, task)
        print('\ncropping...')
        my_crop(ff, task)
