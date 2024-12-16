import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils.sen_spec import sen_spec
from sklearn.metrics import f1_score
from train import LoadData_CLAM, DPPA


def predict(args, ii, model, device):
    model.to(device)
    test_data = LoadData_CLAM(args.test_txt, False)
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

            out, score = model(feature_ct.to(device), feature_rs.to(device), clust_map.to(device))
            out = softmax(out)
            label = label.squeeze()
            predict_y = torch.max(out, dim=1)[1]
            pre_list.append(predict_y)
            label_list.append(label)
            score = score.squeeze().cpu()
            _, indices = torch.sort(score, descending=False)
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

    test_anna1 = f'>>test_acc: {test_acc :.4f}  test_auc: {test_auc :.4f} test_f1:{f1score:.4f}'
    test_anna2 = f'class 0: acc {TN / (TN + FP) :.4f}, correct {TN}/{TN + FP}'
    test_anna3 = f'class 1: acc {TP / (TP + FN) :.4f}, correct {TP}/{TP + FN}'
    print(test_anna1)
    print(test_anna2)
    print(test_anna3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--model', default='DPPA', type=str)
    args = parser.parse_args()
    for file in ['IDH']:
        for ii in range(args.k):
            args.test_txt = f'data_label/{file}/fold_{ii}/test.txt'
            print(f'-----------fold{ii}--------------')
            model = DPPA()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.eval_path = 'eval/' + file
            predict(args, ii, model, device)
