import os
import argparse
import numpy as np
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.pytorchtools import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from utils.sen_spec import sen_spec, curve_plot
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import f1_score
from models.model_clam_20220301 import initialize_weights, Attn_Net_Gated
from torchvision import models


def get_feature_path(txt_path):
    with open(txt_path, 'r', encoding='GBK') as f:
        fea_info = f.readlines()
        fea_info = list(map(lambda x: x.strip().split('\t'), fea_info))
    return fea_info  # 返回图片信息


transform_train = transforms.Compose([
    transforms.Resize([448, 448]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize([448, 448]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LoadData_CLAM(Dataset):
    def __init__(self, args, txt_path, train=False):
        self.data = get_feature_path(txt_path)
        self.train = train
        self.transform_train = transform_train
        self.transform_test = transform_test

        slides = [s[0] for s in self.data]
        self.fea_all, self.fea_center, self.random_index, self.class_ratio, self.epoch, self.map = {}, {}, {}, {}, {}, {}
        for slide in slides:
            self.epoch[slide] = 0
            s = slide.split('.h5')[0]
            with h5py.File('../0_Extracted_feature/' + slide, 'r') as hdf5_file:
                self.fea_all[slide] = hdf5_file['features'][:]

            with h5py.File('../0_Cluster_data/features_100/' + slide, 'r') as hdf5_file:
                self.fea_center[slide] = hdf5_file['features'][:]

            with h5py.File('../0_Random_index/random_sample_all/' + slide, 'r') as hdf5_file:
                self.random_index[slide] = hdf5_file['index'][:]

            img = Image.open(f'../SRIQ/sort_by_p_value/{args.tumor}/{s}.jpg').convert('RGB')
            self.map[slide] = self.transform_train(img) if self.train else self.transform_test(img)

    def __getitem__(self, index):  # 返回真正想返回的东西
        slide, label = self.data[index]
        label = torch.tensor([int(i) for i in label])

        if self.train:
            fea = self.fea_center[slide]
            fea = torch.from_numpy(fea)

            random_index = np.int16(self.random_index[slide][self.epoch[slide]])
            fea1 = self.fea_all[slide][random_index, :]
            fea1 = torch.from_numpy(fea1)

            self.epoch[slide] += 1
        else:
            fea = self.fea_center[slide]
            fea = torch.from_numpy(fea)
            fea1 = self.fea_center[slide]
            fea1 = torch.from_numpy(fea1)

        img = self.map[slide]

        return fea, fea1, img, label

    def __len__(self):
        return len(self.data)


class DPPA(nn.Module):
    def __init__(self, dropout=0, drop_att=0, size1=512):
        super(DPPA, self).__init__()
        self.size1 = size1
        size = [768, self.size1, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        size1 = [768, self.size1, 256]
        fc1 = [nn.Linear(size1[0], size1[1]), nn.LayerNorm(size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net1 = Attn_Net_Gated(L=size1[1], D=size1[2], dropout=drop_att, n_classes=1)
        fc1.append(attention_net1)
        self.attention_net1 = nn.Sequential(*fc1)

        self.resnet = nn.Sequential(*list(models.resnet18().children())[:-1])
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classifiers_1 = nn.Linear(512, 2)
        self.classifiers_2 = nn.Linear(512, 2)
        self.classifiers = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))

        initialize_weights(self)

    def forward(self, h0, h1, map_fea):
        A1, h_1 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        M1 = torch.squeeze(torch.bmm(A1, h_1)).reshape(-1, self.size1)

        A, h = self.attention_net(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A = torch.transpose(A, 2, 1)  # KxN A(8, 1, 100)
        A = F.softmax(A, dim=2)  # softmax over N

        idx = torch.topk(A, int(A.shape[-1] * 0.5), dim=2, largest=True)[1]
        idx = idx.squeeze()
        aa = torch.arange(len(A)).unsqueeze(1)
        h1 = h1[aa, idx, :]

        idx = torch.topk(A1, int(A1.shape[-1] * 0.5), dim=2, largest=True)[1]
        idx = idx.squeeze()
        aa = torch.arange(len(A1)).unsqueeze(1)
        h1 = torch.cat((h0[aa, idx, :], h1), dim=1)

        A2, h2 = self.attention_net1(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A2 = torch.transpose(A2, 2, 1)  # KxN A(8, 1, 100)
        A2 = F.softmax(A2, dim=2)  # softmax over N
        M2 = torch.squeeze(torch.bmm(A2, h2)).reshape(-1, self.size1)

        map_out = self.drop(self.relu(self.resnet(map_fea)))
        map_out = torch.flatten(map_out, 1)

        M = torch.cat((M2, map_out), dim=1)
        logits = self.classifiers(M)
        logits1 = self.classifiers_1(M1)
        logits2 = self.classifiers_2(map_out)
        return logits, logits1, logits2


def predict(args, ii, model, device):
    model.to(device)

    with open(args.test_txt, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        slide_list = list(map(lambda x: x.strip().split('\t')[0].split('\\')[-1].split('.')[0], imgs_info))

    test_data = LoadData_CLAM(args, args.test_txt)
    test_num = len(test_data)
    test_loader = DataLoader(dataset=test_data, num_workers=0, pin_memory=True, batch_size=1)
    test_steps = len(test_loader)

    weight_path = os.path.join(args.eval_path, f'weight_fold_{ii}.pth')
    assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    pre_list = []
    label_list = []
    test_acc_list = []
    test_auc_list = []
    teat_acc_num = 0
    softmax = nn.Softmax(dim=1)
    pro_list = torch.tensor([]).to(device)
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            feature_ct, feature_rs, clust_map, label = data

            out, out1, out2 = model(feature_ct.to(device), feature_rs.to(device), clust_map.to(device))
            out = softmax(out)
            label = label.flatten()
            predict_y = torch.max(out, dim=1)[1]
            pre_list.append(predict_y)
            label_list.append(label)
            pro_list = torch.cat([pro_list, out], dim=0)
            teat_acc_num += torch.eq(predict_y.to(device), label.to(device)).sum().item()

            print(f'\rtest---batch_num: {step}/{test_steps}', end='')
    print()
    pre_list = torch.tensor([k for k in pre_list], device='cpu')
    label_list = torch.tensor([k for k in label_list], device='cpu')
    f1score = f1_score(label_list, pre_list, average='weighted')
    TP, TN, FP, FN = sen_spec(label_list, pre_list)

    test_acc = teat_acc_num / test_num
    test_acc_list.append(test_acc)
    test_auc = roc_auc_score(label_list, pro_list[:, 1].cpu())
    test_auc_list.append(test_auc)

    test_anna1 = f'>>test_acc: {test_acc :.4f}  test_auc: {test_auc :.4f} f1_score: {f1score:.4f}'
    test_anna2 = f'class 0: acc {TN / (TN + FP) :.4f}, correct {TN}/{TN + FP}'
    test_anna3 = f'class 1: acc {TP / (TP + FN) :.4f}, correct {TP}/{TP + FN}'
    print(test_anna1)
    print(test_anna2)
    print(test_anna3)

    result = {'slide': slide_list,
              'label': label_list.tolist(),
              'pre_label': pre_list.tolist(),
              'p_0': pro_list[:, 0].tolist(),
              'p_1': pro_list[:, 1].tolist()}
    result = pd.DataFrame(result)
    result = result.dropna()

    result.to_excel(os.path.join(args.eval_path, f'fold_{ii}.xlsx'))

    return test_acc, test_auc, f1score, (test_anna1, test_anna2, test_anna3)


def train(args, settings, ii):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = LoadData_CLAM(args, args.train_txt, train=True)
    val_data = LoadData_CLAM(args, args.val_txt)
    batch_size = args.batchsize

    train_loader = DataLoader(dataset=train_data, num_workers=0, pin_memory=True, batch_size=batch_size)
    validate_loader = DataLoader(dataset=val_data, num_workers=0, pin_memory=True, batch_size=1)
    train_num = len(train_data)
    val_num = len(val_data)
    net = args.net.to(device)

    weight_path = "model_train/resnet18-f37072fd.pth"
    net.resnet.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    loss_function = nn.CrossEntropyLoss(weight=args.weight_CE.to(device))

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = 0

    sensitivity = []
    specificity = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_auc_list = []
    result = {}

    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    print('batch_total:', train_steps)

    early_stopping = EarlyStopping(patience=20, verbose=True,
                                   path=os.path.join(args.eval_path, f'weight_fold_{ii}.pth'))

    for epoch in range(args.epochs):
        time1 = time.time()
        print(f'Epoch {epoch + 1}/{args.epochs} ', end=' ')
        net.train()
        train_acc_num = 0
        train_loss = 0
        pre_list = []
        label_list = []

        for step, data in enumerate(train_loader):
            feature_ct, feature_rs, clust_map, label = data

            optimizer.zero_grad()
            out, out1, out2 = net(feature_ct.to(device), feature_rs.to(device), clust_map.to(device))
            label = label.flatten()
            loss0 = loss_function(out, label.to(device))
            loss1 = loss_function(out1, label.to(device))
            loss2 = loss_function(out2, label.to(device))
            loss = loss0 + args.bb * loss1 + args.cc * loss2
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predict_y = torch.max(out, dim=1)[1]
            pre_list.append(predict_y)
            label_list.append(label)

            train_acc_num += torch.eq(predict_y.to(device), label.to(device)).sum().item()

        pre_list = [i for k in pre_list for i in k]
        label_list = [i for k in label_list for i in k]
        TP, TN, FP, FN = sen_spec(label_list, pre_list)

        train_loss_list.append(train_loss / train_steps)
        train_accurate = train_acc_num / train_num
        train_acc_list.append(train_accurate)

        train_anna1 = f't_loss:{train_loss / train_steps :.4f} t_acc:{train_accurate:.4f}'
        train_anna2 = f'class 0: acc {TN / (TN + FP) :.4f}, correct {TN}/{TN + FP}'
        train_anna3 = f'class 1: acc {TP / (TP + FN) :.4f}, correct {TP}/{TP + FN}'
        print(train_anna1, end='  ')

        net.eval()
        val_acc_num = 0
        val_loss = 0
        pre_list = []
        label_list = []
        pro_list = torch.tensor([]).to(device)
        with torch.no_grad():
            for val_data in validate_loader:
                feature_ct, feature_rs, clust_map, label = val_data

                out, out1, out2 = net(feature_ct.to(device), feature_rs.to(device), clust_map.to(device))
                label = label.flatten()

                loss = loss_function(out, label.to(device))
                pro = nn.Softmax(dim=1)(out)
                predict_y = torch.max(out, dim=1)[1]
                pre_list.append(predict_y)
                label_list.append(label)
                pro_list = torch.cat([pro_list, pro], dim=0)

                val_acc_num += torch.eq(predict_y.to(device), label.to(device)).sum().item()
                val_loss += loss.item()

        pre_list = torch.tensor([i for k in pre_list for i in k])
        label_list = torch.tensor([i for k in label_list for i in k])
        f1score = f1_score(label_list, pre_list, average='weighted')
        TP, TN, FP, FN = sen_spec(label_list, pre_list)
        sensitivity.append(TP / (TP + FN))
        specificity.append(TN / (TN + FP))

        val_loss = val_loss / val_steps
        val_loss_list.append(val_loss)
        val_accurate = val_acc_num / val_num
        val_acc_list.append(val_accurate)
        val_auc = roc_auc_score(label_list, pro_list[:, 1].cpu())
        val_auc_list.append(val_auc)

        val_anna1 = f'v_acc:{val_accurate :.4f}  v_auc:{val_auc :.4f}  f1:{f1score:.4f}  v_loss:{val_loss:.4f}'
        val_anna2 = f'class 0: acc {TN / (TN + FP) :.4f}, correct {TN}/{TN + FP}'
        val_anna3 = f'class 1: acc {TP / (TP + FN) :.4f}, correct {TP}/{TP + FN}'
        print(val_anna1, end='  ')
        epoch_time = time.time() - time1
        print(f'用时 {epoch_time:0.1f} s', end=" ")

        record = early_stopping(val_accurate + val_auc, net, epoch)
        if record:
            best_epoch = epoch + 1
            settings['best_epoch'] = best_epoch
            result['train'] = (train_anna1, train_anna2, train_anna3)
            result['val'] = (val_anna1, val_anna2, val_anna3)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    curve_plot(train_loss_list, val_loss_list, val_auc_list, train_acc_list, val_acc_list, args, ii)

    print('best_epoch: ', best_epoch)
    test_acc, test_auc, test_f1, test_anna = predict(args, ii, net, device)
    ES_acc.append(test_acc)
    ES_auc.append(test_auc)
    ES_f1.append(test_f1)
    result['test'] = test_anna

    settings.update(result)
    ES_acc_mean, ES_auc_mean, ES_f1_mean = np.array(ES_acc).mean(), np.array(ES_auc).mean(), np.array(ES_f1).mean()
    with open(os.path.join(args.eval_path, f'result_fold_{ii}.txt'), 'w') as f:
        for key, value in settings.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')
        f.write(f'ES_mean_acc:{ES_acc_mean} \n')
        f.write(f'ES_mean_auc:{ES_auc_mean} \n')
        f.write(f'ES_mean_f1:{ES_f1_mean} \n')
    if ii == 4:
        with open(os.path.join(args.eval_path, f'aa_{ES_acc_mean:.4f}_{ES_auc_mean:.4f}.txt'), 'w') as f:
            f.write(f'ES_mean_acc:{ES_acc_mean} \n')
            f.write(f'ES_mean_auc:{ES_auc_mean} \n')
            f.write(f'ES_mean_f1:{ES_f1_mean} \n')
    print('Finished Training')


def seed_torch(seed=1):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for my_model Training')
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--drop_att', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--cos_lr', default=False, type=bool)
    parser.add_argument('--batchsize', default=32, type=str)
    parser.add_argument('--seed', default=2023, type=str)
    parser.add_argument('--bb', default=0.1, type=float)
    parser.add_argument('--cc', default=0.1, type=float)
    args = parser.parse_args()

    for file in ['IDH', 'Grading']:
        args.tumor = file
        ES_acc, ES_auc, ES_f1 = [], [], []
        for ii in range(args.k):
            seed_torch(args.seed)
            args.net = DPPA(args.drop, args.drop_att, 512)

            args.train_txt = f'data_label/{file}/fold_{ii}/train.txt'
            args.val_txt = f'data_label/{file}/fold_{ii}/val.txt'
            args.test_txt = f'data_label/{file}/fold_{ii}/test.txt'

            args.eval_path = 'eval/' + file

            os.makedirs(args.eval_path, exist_ok=True)

            settings = {'model': args.model, 'dropout': args.drop, 'k': args.k, 'epochs': args.epochs,
                        'lr': args.lr, 'cos_lr': args.cos_lr, 'loss': args.loss, 'weight': args.weight_CE,
                        'eval_path': args.eval_path, 'seed': args.seed}

            train(args, settings, ii)
        if len(ES_acc) > 0:
            print('ES_mean_acc:', np.array(ES_acc).mean())
            print('ES_mean_auc:', np.array(ES_auc).mean())
            print('ES_mean_f1:', np.array(ES_f1).mean())
