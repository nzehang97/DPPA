import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch_geometric.nn import GCNConv,GraphConv,Linear,GATConv,GraphSAGE
from sklearn.preprocessing import StandardScaler
import torch_geometric.nn as pyg_nn
from resnest.torch import resnest
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from torch.autograd import Variable
from functools import reduce
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


from torch_geometric.utils import degree
from torch_geometric.nn.pool.topk_pool import filter_adj, topk



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        # self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        # for i in range(M):
        #     # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
        #     self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
        #                                    nn.BatchNorm2d(out_channels),
        #                                    nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
        self.avgpool = GlobalAvgPool2d()
        self.lin = nn.Linear(128, 2)
    def forward(self, out1,out2,out3):
        map1, map2, map3 = self.to_featuremap(out1, out2, out3)
        batch_size=map1.size(0)
        output=[]
        output.append(map1)
        output.append(map2)
        output.append(map3)
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加         #(batchsize*128*16*16))
        V = self.avgpool(V)
        V = torch.flatten(V, 1)
        V=self.lin(V)
        return V
    def to_featuremap(self,out1,out2,out3):
        batch_size=out1.shape[0]
        len=out1.shape[1]
        map_size1=out3.shape[2]
        map_size2 = out3.shape[3]
        map1=torch.zeros((batch_size,len,map_size1,map_size2)).to(device)
        map2=torch.zeros((batch_size,len,map_size1,map_size2)).to(device)
        for i in range(batch_size):
            for j in range(len):
                map1[i,j,:,:]=out1[i,j]
                map2[i, j, :, :] = out2[i, j]
        map3=out3
        return map1,map2,map3

class low_rank_fusion(torch.nn.Module):
    def __init__(self,net1_out,net2_out,net3_out,num_classes=2):
        super(low_rank_fusion, self).__init__()
        self.rank = 4
        self.viewA_factor = Parameter(torch.Tensor(self.rank, net1_out + 1, num_classes))
        self.viewB_factor = Parameter(torch.Tensor(self.rank, net2_out + 1, num_classes))
        #self.viewC_factor = Parameter(torch.Tensor(self.rank, net3_out + 1, num_classes))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, num_classes))
        # init teh factors
        xavier_normal(self.viewA_factor)
        xavier_normal(self.viewB_factor)
        #xavier_normal(self.viewC_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    def forward(self, x1, x2):
        batch_size = x1.data.shape[0]

        if x1.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _viewA_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), x1), dim=1)
        _viewB_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), x2), dim=1)
        #_viewC_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), x3), dim=1)

        fusion_viewA = torch.matmul(_viewA_h, self.viewA_factor)
        fusion_viewB = torch.matmul(_viewB_h, self.viewB_factor)
        #fusion_viewC = torch.matmul(_viewC_h, self.viewC_factor)
        fusion_zy = fusion_viewA * fusion_viewB #* fusion_viewC
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = F.softmax(output)
        return output

class three_model(torch.nn.Module):
    def __init__(self, net1_in,net2_in, net1_hid,net2_hid,net1_out,net2_out, out_feats,net3_in=1):
        super(three_model, self).__init__()

        self.net1=GCN(net1_in, net1_hid, net1_out)
        self.net2=GAT(net2_in, net2_hid, net2_out)
        self.net3=resnest.resnest18(net3_in)
        #self.net3=resnet_wu.resnet18(net3_in)

        self.mlp1 = mlp(net1_out, out_feats)
        #self.mlp2 = mlp(int((net1_out+net2_out+128)/3), out_feats)
        self.low_rank_fusion = low_rank_fusion(net1_out=128, net2_out=128, net3_out=2048)
        self.SKConv = SKConv(128,128,stride=1,M=3,r=16,L=32)


    def forward(self, data):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        data3 = data[2]['grey'].unsqueeze(axis=1).to(device)

        out1 = self.net1(data1)           #8*128
        #out2 = self.net2(data2)           #8*128
        #out3 = self.net3(data3)           #8*2048

        #mlp_input=torch.cat((out1,out2,out3),1)
        #mlp_input=out2
        #result=self.mlp1(mlp_input)
        #result = self.mlp2(mlp_input)
        #fusion_result = self.SKConv(out1, out2, out3)
        #fusion_result =self.low_rank_fusion(out1,out2)

        return out1


class mlp(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(mlp, self).__init__()
        self.lin = Linear(in_feats, out_feats)
    def forward(self, x):
        x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)  # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        self.conv2 = GCNConv(hidden_feats, out_feats)  # 构造第二层，输入和输出通道，输出通道的大小和图或者节点的分类数量一致，比如此程序中图标记就是二分类0和1，所以等于2
        self.lin = Linear(hidden_feats, out_feats)
        #self.lin = Linear(32, 2)
    def forward(self, data):  # 前向传播
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 赋值
        x=torch.from_numpy(StandardScaler().fit_transform(x.cpu()))
        x = x.to(torch.float32).to(device)
        x1 = self.conv1(x, edge_index)  # 第一层启动运算，输入为节点及特征和边的稀疏矩阵，输出结果是二维度[20张图的所有节点数,128]
        # print(x.shape)
        x1 = F.relu(x1)  # 激活函数
        x1 = F.dropout(x1, training=self.training)
        x2 = self.conv2(x1, edge_index)  # 第二层启动运算，输入为节点及特征和边的稀疏矩阵，输出结果是二维度[20张图的所有节点数,2]
        # print(batch) # 每张图片的节点的分类值不同0-19（0-19这个范围大小是根据batch_size的数目更新）
        x2 = pyg_nn.global_mean_pool(x2,batch)  # 池化降维，根据batch的值知道有多少张图片（每张图片的节点的分类值不同0-19），再将每张图片的节点取一个全局最大的节点作为该张图片的一个输出值
        # print(x.shape) # 输出维度变成[20，2]
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x3 = self.lin(x2)
        a = F.log_softmax(x3, dim=1)  # softmax可以得到每张图片的概率分布，设置dim=1，可以看到每一行的加和为1，再取对数矩阵每个结果的值域变成负无穷到0
        return x3

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GAT, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.gat1 = GATConv(in_feats, hidden_feats, heads=2)
        self.conv2 = GraphConv(hidden_feats * 3, out_feats)

        self.lin = Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.from_numpy(StandardScaler().fit_transform(x.cpu()))
        x = x.to(torch.float32).to(device)
        # 第一层GraphConv
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        #x1 = F.dropout(x1, p=0.6, training=self.training) #2560,256

        # 第一层GATConv
        x_gat1 = self.gat1(x, edge_index)
        x_gat1 = x_gat1.relu()
        #x_gat1 = F.dropout(x_gat1, p=0.6, training=self.training) #2560,512

        # 拼接GraphConv和GATConv的结果
        x2 = torch.cat([x1, x_gat1], dim=1)#2560,768

        # 第3层GraphConv
        x3 = self.conv2(x2, edge_index)
        x3 = pyg_nn.global_mean_pool(x3, batch)
        x3 = F.dropout(x3, p=0.5, training=self.training)

        # 全连接层
        #x3 = self.lin(x3)
        #x3=F.log_softmax(x3, dim=1)
        return x3

class GraphConvSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GraphConvSAGE, self).__init__()

        self.num_layers = num_layers

        # GraphConv 层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConv(input_dim, hidden_dim))  # 输入层

        for _ in range(num_layers - 1):
            self.conv_layers.append(GraphConv(hidden_dim, hidden_dim))  # 隐藏层

        # GraphSAGE 层
        self.sage = GraphSAGE(hidden_dim, hidden_dim, num_layers)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.from_numpy(StandardScaler().fit_transform(x))
        x = x.to(torch.float32)
        # GraphConv 层
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
        # GraphSAGE 层
        x = self.sage(x, edge_index)

        # 全局平均池化
        # x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))#torch.zeros(x.size(0), dtype=torch.long)作为虚拟的批次向量，表示所有节点属于同一个图或批次
        x = pyg_nn.global_mean_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        # 输出层
        x = self.fc(x)
        return x

class MVPool_GCN(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)  # 构造第一层，输入和输出通道，输入通道的大小和节点的特征维度一致
        self.conv2 = GCNConv(hidden_feats, hidden_feats)  # 构造第二层，输入和输出通道，输出通道的大小和图或者节点的分类数量一致，比如此程序中图标记就是二分类0和1，所以等于2
        self.lin = Linear(hidden_feats*2, out_feats)
        self.pool = MVPool()
    def forward(self, data):  # 前向传播
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        x=torch.from_numpy(StandardScaler().fit_transform(x.cpu()))
        x = x.to(torch.float32).to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _ = self.pool(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _ = self.pool(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2)
        x = F.relu(self.lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(x, dim=-1)

        return x

class MVPool(torch.nn.Module):
    def __init__(self, in_channels, ratio):
        super(MVPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.weight = Parameter(torch.Tensor(1, in_channels))
        nn.init.xavier_uniform_(self.weight.data)
        self.view_att = Parameter(torch.Tensor(3, 3))
        nn.init.xavier_uniform_(self.view_att.data)
        self.alpha = Parameter(torch.Tensor(1))
        nn.init.ones_(self.alpha.data)

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        row, col = edge_index
        score1 = torch.sigmoid(self.alpha * torch.log(degree(row, num_nodes=x.size(0)) + 1e-16) + self.beta).view(-1, 1)
        x_score2 = (x * self.weight).sum(dim=-1)
        score2 = torch.sigmoid(x_score2 / self.weight.norm(p=2, dim=-1)).view(-1, 1)
        x_score3 = self.calc_pagerank_score(x, edge_index, edge_attr)
        score3 = torch.sigmoid(x_score3).view(-1, 1)

        score_cat = torch.cat([score1, score2, score3], dim=1)
        max_value, _ = torch.max(torch.abs(score_cat), dim=0)
        score_cat = score_cat / max_value
        score_weight = torch.sigmoid(torch.matmul(score_cat, self.view_att) + self.view_bias)
        score_weight = torch.softmax(score_weight, dim=1)
        score = torch.sigmoid(torch.sum(score_cat * score_weight, dim=1))
        # score = score2.view(-1)

        # Graph Pooling
        original_x = x
        perm = topk(score, self.ratio, batch)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # Discard structure learning layer, directly return
        return x, induced_edge_index, induced_edge_attr, batch, perm





