import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from utils.sen_spec import sen_spec
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import f1_score as sklearn_f1_score
import models.dsmil as mil


trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout != 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def get_feature_path(txt_path):
    with open(txt_path, 'r', encoding='GBK') as f:
        fea_info = f.readlines()
        fea_info = list(map(lambda x: x.strip().split('\t'), fea_info))
    return fea_info  # 返回图片信息


class LoadData_CLAM1(Dataset):
    def __init__(self, txt_path, train=False):
        self.data = get_feature_path(txt_path)
        self.train = train
        self.roi_transforms = trnsfrms_val
        slides = [s[0] for s in self.data]
        self.fea_all = {}
        for slide in slides:
            with h5py.File('../0_Extracted_feature/' + slide, 'r') as hdf5_file:
                feature = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
                self.fea_all[slide] = feature

    def __getitem__(self, index):  # 返回真正想返回的东西
        slide, label = self.data[index]
        label = torch.tensor([int(i) for i in label])

        Fea = self.fea_all[slide]
        Fea = torch.from_numpy(Fea)
        return Fea, label

    def __len__(self):
        return len(self.data)


def predict(args, ii, model, device):
    model.to(device)

    test_data = LoadData_CLAM1(args.test_txt)
    test_num = len(test_data)
    test_loader = DataLoader(dataset=test_data, num_workers=0, pin_memory=True, batch_size=1)
    test_steps = len(test_loader)
    weight_path = os.path.join(args.weight_path, args.task, f'DSMIL/weight_fold_{ii}.pth')
    assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    pre_list = []
    label_list = []
    test_acc_list = []
    test_auc_list = []
    teat_acc_num = 0
    pro_list = torch.tensor([]).to(device)
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            feature_ct, label = data
            label = label.flatten()
            ins_prediction, bag_prediction, _, _ = model(feature_ct.to(device))
            max_prediction, _ = torch.max(ins_prediction, 1)
            out = nn.Softmax(dim=1)((bag_prediction+max_prediction)/2)
            predict_y = torch.max(bag_prediction+max_prediction, dim=1)[1]
            pre_list.append(predict_y)
            label_list.append(label)
            pro_list = torch.cat([pro_list, out], dim=0)
            teat_acc_num += torch.eq(predict_y.to(device), label.to(device)).sum().item()

            print(f'\rtest---batch_num: {step}/{test_steps}', end='')
    print()
    pre_list = torch.tensor([i for k in pre_list for i in k], device='cpu')
    label_list = torch.tensor([i for k in label_list for i in k], device='cpu')
    f1score = sklearn_f1_score(label_list, pre_list, average='binary')
    TP, TN, FP, FN = sen_spec(label_list, pre_list)

    test_acc = teat_acc_num / test_num
    test_acc_list.append(test_acc)
    test_auc = roc_auc_score(label_list, pro_list[:, 1].cpu())
    test_auc_list.append(test_auc)

    test_anna1 = f'>>test_acc: {test_acc :.4f}  test_auc: {test_auc :.4f}  f1score: {f1score:.4f}'
    test_anna2 = f'class 0: acc {TN / (TN + FP) :.4f}, correct {TN}/{TN + FP}'
    test_anna3 = f'class 1: acc {TP / (TP + FN) :.4f}, correct {TP}/{TP + FN}'
    print(test_anna1)
    print(test_anna2)
    print(test_anna3)

    return test_acc, test_auc, f1score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--task', default='Grading', type=str)
    parser.add_argument('--weight_path', default='./weight', type=str)
    args = parser.parse_args()

    all_test_acc = []
    all_test_auc = []
    all_f1_score = []

    for ii in range(args.k):
        args.test_txt = f'../model_train/data_label/{args.task}/fold_{ii}/test.txt'
        print(f'-----------fold{ii}--------------')
        i_classifier = mil.FCLayer(in_size=768, out_size=2)
        b_classifier = mil.BClassifier(input_size=768, output_class=2, dropout_v=0, nonlinear=1)
        model = mil.MILNet(i_classifier, b_classifier)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_acc, test_auc, f1_score = predict(args, ii, model, device)  # 获取这一轮的结果
        all_test_acc.append(test_acc)
        all_test_auc.append(test_auc)
        all_f1_score.append(f1_score)

    # 计算平均值和标准差
    mean_test_acc = np.mean(all_test_acc)
    std_test_acc = np.std(all_test_acc)
    mean_test_auc = np.mean(all_test_auc)
    std_test_auc = np.std(all_test_auc)
    mean_f1_score = np.mean(all_f1_score)
    std_f1_score = np.std(all_f1_score)

    print(f"平均test_acc: {mean_test_acc:.4f}，标准差: {std_test_acc:.4f}")
    print(f"平均test_auc: {mean_test_auc:.4f}，标准差: {std_test_auc:.4f}")
    print(f"平均f1_score: {mean_f1_score:.4f}，标准差: {std_f1_score:.4f}")
