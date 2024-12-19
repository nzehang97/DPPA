import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from utils.sen_spec import sen_spec
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import f1_score as sklearn_f1_score

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
        return Fea, 0, 0, label

    def __len__(self):
        return len(self.data)


class CLAM_SB(nn.Module):
    def __init__(self, dropout=0, drop_att=0, n_classes=2):
        super(CLAM_SB, self).__init__()
        size = [768, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.attention_net1 = nn.Sequential(*fc)
        self.classifiers_1 = nn.Sequential(nn.Linear(size[1] * 2, 2))
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.subtyping = False
        self.n_classes = n_classes
        self.k_sample = 8
        initialize_weights(self)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h0, h1=0, label=None, instance_eval=True):
        A, h = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A = torch.transpose(A, 2, 1)  # KxN A(8, 1, 100)
        A = F.softmax(A, dim=2)  # softmax over N

        idx = torch.topk(A, A.shape[-1] // 2, dim=2, largest=True)[1]
        idx = idx.flatten()
        h1 = h0[:, idx, :]

        A1, h1 = self.attention_net1(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h.squeeze(), classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h.squeeze(), classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
        M1 = torch.squeeze(torch.bmm(A, h), axis=1)
        M2 = torch.squeeze(torch.bmm(A1, h1), axis=1)
        M = torch.cat((M1, M2), dim=1)
        logits1 = self.classifiers_1(M)

        return logits1, total_inst_loss


def predict(args, ii, model, device):
    model.to(device)

    test_data = LoadData_CLAM1(args.test_txt)
    test_num = len(test_data)
    test_loader = DataLoader(dataset=test_data, num_workers=0, pin_memory=True, batch_size=1)
    test_steps = len(test_loader)
    weight_path = os.path.join(args.weight_path, args.task, f'CLAM_SB/weight_fold_{ii}.pth')
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
            label = label.flatten()
            out, _ = model(feature_ct.to(device), feature_rs.to(device), label.to(device))
            out = softmax(out)

            predict_y = torch.max(out, dim=1)[1]
            pre_list.append(predict_y)
            label_list.append(label)
            pro_list = torch.cat([pro_list, out], dim=0)
            teat_acc_num += torch.eq(predict_y.to(device), label.to(device)).sum().item()

            print(f'\rtest---batch_num: {step}/{test_steps}', end='')
    print()
    pre_list = torch.tensor([k for k in pre_list], device='cpu')
    label_list = torch.tensor([k for k in label_list], device='cpu')
    f1score = sklearn_f1_score(label_list, pre_list, average='weighted')
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
        model = CLAM_SB()
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
