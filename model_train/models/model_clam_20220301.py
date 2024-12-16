import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet import Bottleneck, ResNet


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


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


class CLAM_SB(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB, self).__init__()
        size = [769, 512, 256]
        # fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers_1 = nn.Linear(size[1]+10, 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LayerNorm([64, 112, 112]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm([128, 56, 56]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LayerNorm([256, 28, 28]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(256, 10, 3, stride=2, padding=1),
            nn.LayerNorm([10, 14, 14]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        initialize_weights(self)

    def forward(self, h0, h1, clstermap):
        # h0 = h0.to(torch.float32)
        A, h = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        AA = torch.transpose(A, 2, 1)  # KxN A(8, 1, 100)
        A = F.softmax(AA, dim=2)  # softmax over N
        M = torch.squeeze(torch.bmm(A, h), axis=1)
        map_out = self.cnn(clstermap)
        map_out = torch.flatten(map_out, 1)

        M = torch.cat((M, map_out), dim=1)
        logits = self.classifiers_1(M)

        return logits, AA


class CLAM_SB_topk(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB_topk, self).__init__()
        size = [768, 512, 256]
        # fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.attention_net = nn.Sequential(*fc)
        self.pooling = nn.MaxPool2d((20, 1))
        self.relu = nn.ReLU()
        self.LN = nn.LayerNorm(size[2])
        self.out = nn.Linear(size[1], size[2])
        self.classifiers_1 = nn.Linear(size[2], 2)
        self.drop = nn.Dropout(0.5)
        initialize_weights(self)

    def forward(self, h0, h1):
        M1 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        M1 = self.drop(self.relu(self.LN(self.out(M1))))
        out = self.pooling(M1).view(M1.size()[0], -1)
        out = torch.tanh(out)

        M2 = self.attention_net(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        M2 = self.drop(self.relu(self.LN(self.out(M2))))
        out_neg = self.pooling(M2).view(M2.size()[0], -1)
        out_neg = torch.tanh(out_neg)
        cos_similarity = torch.cosine_similarity(out, out_neg, dim=1)
        logits = self.classifiers_1(self.drop(out))

        return logits, 0, cos_similarity


class CLAM_SB_multi(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB_multi, self).__init__()
        size = [512, 256, 128]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        fc1 = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net1 = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc1.append(attention_net1)
        self.attention_net1 = nn.Sequential(*fc1)

        self.classifiers_1 = nn.Linear(size[1] * 2, 2)

        initialize_weights(self)

    def forward(self, h0, h1):
        h0 = h0.to(torch.float32)
        h1 = h1.to(torch.float32)
        A0, h0 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A0 = torch.transpose(A0, 2, 1)  # KxN A(8, 1, 100)
        A0 = F.softmax(A0, dim=2)  # softmax over N
        M0 = torch.squeeze(torch.bmm(A0, h0), axis=1)

        A1, h1 = self.attention_net1(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        M1 = torch.squeeze(torch.bmm(A1, h1), axis=1)
        M = torch.concat((M0, M1), dim=1)
        # M = (M0+M1)/2
        # M = M0

        logits = self.classifiers_1(M)

        return logits


class CLAM_SB_Con(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB_Con, self).__init__()
        size = [769, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers_1 = nn.Sequential(nn.Linear(size[1], 2))
        self.classifiers_2 = nn.Sequential(nn.Linear(size[1], 2))
        initialize_weights(self)

    def forward(self, h0, h1):
        A1, h_1 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        M1 = torch.squeeze(torch.bmm(A1, h_1), axis=1)

        A2, h_2 = self.attention_net(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A2 = torch.transpose(A2, 2, 1)  # KxN A(8, 1, 100)
        A2 = F.softmax(A2, dim=2)  # softmax over N
        M2 = torch.squeeze(torch.bmm(A2, h_2), axis=1)

        logits1 = self.classifiers_1(M1)
        logits2 = self.classifiers_2(M2)

        return logits1, logits2


class CLAM_SB_Con_cat(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB_Con_cat, self).__init__()
        size = [768, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers_1 = nn.Sequential(nn.Linear(size[1]*2+10, 2))
        self.classifiers_2_1 = nn.Linear(size[1]*2+10, 20)
        self.nn_20 = nn.Sequential(nn.LayerNorm(20), nn.ReLU(), nn.Dropout(dropout))
        self.nn_40 = nn.Sequential(nn.LayerNorm(40), nn.ReLU(), nn.Dropout(dropout))
        self.classifiers_2_2 = nn.Linear(20, 20)
        self.classifiers_2_3 = nn.Linear(40+size[1]*2+10, 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LayerNorm([64, 112, 112]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LayerNorm([128, 56, 56]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LayerNorm([256, 28, 28]),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(256, 10, 3, stride=2, padding=1),
            nn.LayerNorm([10, 14, 14]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        initialize_weights(self)

    def forward(self, h0, h1, map):
        A1, h_1 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        M1 = torch.squeeze(torch.bmm(A1, h_1), axis=1)

        A2, h_2 = self.attention_net(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A2 = torch.transpose(A2, 2, 1)  # KxN A(8, 1, 100)
        A2 = F.softmax(A2, dim=2)  # softmax over N
        M2 = torch.squeeze(torch.bmm(A2, h_2), axis=1)

        map_out = self.cnn(map)
        map_out = torch.flatten(map_out, 1)

        M = torch.cat((M1, M2, map_out), dim=1)
        logits1 = self.classifiers_1(M)
        # x = self.classifiers_2_1(M)
        # x1 = self.classifiers_2_2(self.nn_20(x))
        # x = torch.cat((x, x1, M), dim=1)

        # logits1 = self.classifiers_2_3(x)
        return logits1, 0


class CLAM_SB_Con_Cox(nn.Module):
    def __init__(self, dropout, drop_att):
        super(CLAM_SB_Con_Cox, self).__init__()
        size = [768, 512, 256]
        fc = [nn.Linear(size[0], size[1]), nn.BatchNorm1d(49), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=drop_att, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers_1 = nn.Sequential(nn.Linear(size[1], 1))
        self.classifiers_2 = nn.Sequential(nn.Linear(size[1], 1))
        initialize_weights(self)

    def forward(self, h0, h1):
        A1, h_1 = self.attention_net(h0[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A1 = torch.transpose(A1, 2, 1)  # KxN A(8, 1, 100)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        M1 = torch.squeeze(torch.bmm(A1, h_1), axis=1)
        # M1 = torch.mean(h, dim=1)

        A2, h_2 = self.attention_net(h1[:, :, :])  # NxK A(8,100,1) h(8,100,512)
        A2 = torch.transpose(A2, 2, 1)  # KxN A(8, 1, 100)
        A2 = F.softmax(A2, dim=2)  # softmax over N
        M2 = torch.squeeze(torch.bmm(A2, h_2), axis=1)
        # M2 = torch.mean(h, dim=1)

        logits1 = self.classifiers_1(M1)
        logits2 = self.classifiers_2(M2)

        return logits1, logits2, 0


class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=512, dim2=512, scale_dim1=4, scale_dim2=4, mmhid=64, dropout_rate=0.5):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out