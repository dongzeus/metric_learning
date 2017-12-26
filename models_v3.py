import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# test for git

class FeatLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128):
        super(FeatLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, out_size)

    def forward(self, feats):
        h_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        c_t = Variable(torch.zeros(feats.size(0), self.hidden_size).float(), requires_grad=False)
        h_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)
        c_t2 = Variable(torch.zeros(feats.size(0), self.out_size).float(), requires_grad=False)

        if feats.is_cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()
            h_t2 = h_t2.cuda()
            c_t2 = c_t2.cuda()

        for _, feat_t in enumerate(feats.chunk(feats.size(1), dim=1)):
            h_t, c_t = self.lstm1(feat_t[:, 0, :], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            if _ == 0:
                stream = h_t2.view(h_t2.size(0), 1, -1)
            else:
                stream = torch.cat((stream, h_t2.view(h_t2.size(0), 1, -1)), dim=1)
        # aggregated feature

        return stream


class VAMetric_conv(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric_conv, self).__init__()

        self.vLSTM = FeatLSTM(1024, 512, 128)
        self.aLSTM = FeatLSTM(128, 128, 128)

        self.vfc1 = nn.Linear(in_features=1024, out_features=512)
        self.vfc2 = nn.Linear(in_features=512, out_features=128)
        self.afc1 = nn.Linear(in_features=128, out_features=128)
        self.afc2 = nn.Linear(in_features=128, out_features=128)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)  # output bn*32*120
        # self.mp = nn.MaxPool1d(kernel_size=4)
        self.dp = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1)  # output bn*32*113
        self.fc3 = nn.Linear(in_features=32 * 113, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1)
        self.fc5 = nn.Linear(in_features=512, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=2)
        self.init_params()

    def init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):

        # vfeat = self.vLSTM(vfeat)
        # afeat = self.aLSTM(afeat)

        vfeat = self.vfc1(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.vfc2(vfeat)
        vfeat = F.relu(vfeat)

        vfeat = vfeat.view(vfeat.size(0), 1, 1, -1)
        afeat = afeat.view(afeat.size(0), 1, 1, -1)

        vafeat = torch.cat((vfeat, afeat), dim=2)
        vafeat = self.conv1(vafeat)
        vafeat = self.dp(vafeat)
        vafeat = vafeat.view(vafeat.size(0), vafeat.size(1), -1)
        vafeat = self.conv2(vafeat)

        vafeat = vafeat.view([vafeat.size(0), -1])
        vafeat = self.fc3(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc4(vafeat)

        result = F.relu(vafeat)

        return result, 0, 0


# only to test the git hub

class conv_loss_dqy(torch.nn.Module):
    def __init__(self):
        super(conv_loss_dqy, self).__init__()

    def forward(self, sim, label):
        length = len(sim)
        loss1 = torch.mean(torch.pow(label * sim, 2))
        loss2 = 1 - torch.mean(torch.pow((1 - label) * sim, 2))
        loss3 = 1 - torch.mean(sim[0:length / 2 - 1] - sim[length / 2:length - 1])
        # loss1 = torch.mean(torch.pow((1 - label) * sim[:, 1], 2))
        # loss2 = torch.mean(torch.pow(label * sim[:, 0], 2))
        # loss3 = 2.2 - (torch.mean(sim[0:length / 2 - 1]) - torch.mean(sim[length / 2:length - 1]))
        # loss3 = 1 - torch.mean(torch.abs(sim[:, 0] - sim[:, 1]))
        return 1 * loss1 + 1 * loss2 + 1 * loss3


#
# class N_pair_loss(torch.nn.Module):
#     def __init__(self):
#         super(N_pair_loss, self).__init__()
#
#     def forward(self, dis, margin=1):
#         bn = dis.size()[0]
#         loss = 0
#         for i in range(bn):
#
#             Dij = dis[i, i]
#             Dik = dis[i, :].clone()
#             Dik[i] = 0
#             Djk = dis[:, i].clone()
#             Djk[i] = 0
#             margin_ = margin * torch.autograd.Variable(torch.ones(Dik.size())).cuda()
#             loss_i = torch.log(torch.sum(torch.exp(margin_ - Dik) + torch.exp(margin_ - Djk), dim=0)) + Dij
#             if torch.norm(loss_i, p=1).data[0] < 0:
#                 continue
#             else:
#                 loss_i = torch.pow(loss_i, 2)
#                 loss = loss + loss_i
#         loss = loss / (2 * bn)
#
#         return loss
#
class N_pair_loss(torch.nn.Module):
    def __init__(self):
        super(N_pair_loss, self).__init__()

    def forward(self, sim_0, sim_1):
        bn = sim_0.size()[0]
        loss1 = torch.mean(torch.diag(sim_1))
        sim_0 = sim_0 - torch.diag(torch.diag(sim_0))
        # loss2 = torch.mean(torch.max(sim,dim=1)[0])
        loss2 = torch.mean(torch.mean(sim_0, dim=1), dim=0)

        return loss1 + loss2
