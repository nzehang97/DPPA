import os
import h5py
import torch
import torch.nn as nn
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from train import DPPA
from torch.utils.data import Dataset
import torchvision.transforms as transforms


transform_test = transforms.Compose([
    transforms.Resize([448, 448]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_feature_path(txt_path):
    with open(txt_path, 'r', encoding='GBK') as f:
        fea_info = f.readlines()
        fea_info = list(map(lambda x: x.strip(), fea_info))
    return fea_info  # 返回图片信息


class LoadData_CLAM(Dataset):
    def __init__(self, args, txt_path, train=False):
        self.data = get_feature_path(txt_path)
        self.train = train
        self.transform_test = transform_test
        slides = self.data
        self.fea_center, self.map = {}, {}
        for slide in slides:
            s = slide.split('.h5')[0]
            with h5py.File('../0_Cluster_data/features_100/' + slide, 'r') as hdf5_file:
                self.fea_center[slide] = hdf5_file['features'][:]

            img = Image.open(f'../SRIQ/map/{args.task}/{s}.jpg').convert('RGB')
            self.map[slide] = self.transform_train(img) if self.train else self.transform_test(img)

    def __getitem__(self, index):  # 返回真正想返回的东西
        slide = self.data[index]

        fea = self.fea_center[slide]
        fea = torch.from_numpy(fea)
        fea1 = fea

        img = self.map[slide]
        return fea, fea1, img

    def __len__(self):
        return len(self.data)


def predict(args, ii, model, device):
    model.to(device)
    test_data = LoadData_CLAM(args, args.test_txt)
    test_loader = DataLoader(dataset=test_data, num_workers=0, pin_memory=True, batch_size=1)

    weight_path = os.path.join(args.weight_path, args.task, f'DPPA/weight_fold_{ii}.pth')
    assert os.path.exists(weight_path), "ff: '{}' dose not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            feature_ct, feature_rs, clust_map = data

            out, out1, out2 = model(feature_ct.to(device), feature_rs.to(device), clust_map.to(device))
            out = softmax(out)
            predict_y = torch.max(out, dim=1)[1]

            print(f'predict:{predict_y}, probability:{out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--task', default='Grading', type=str)
    parser.add_argument('--model', default='DPPA', type=str)
    parser.add_argument('--weight_path', default='./weight', type=str)
    parser.add_argument('--test_txt', default='./test.txt', type=str)
    args = parser.parse_args()

    for ii in range(args.k):
        print(f'-----------fold{ii}--------------')
        model = DPPA()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predict(args, ii, model, device)
