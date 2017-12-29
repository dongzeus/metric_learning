import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class VAMetric_conv(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric_conv, self).__init__()

        # self.vLSTM = FeatLSTM(1024, 512, 128)
        # self.aLSTM = FeatLSTM(128, 128, 128)

        self.vfc1 = nn.Linear(in_features=1024, out_features=512)
        self.vfc2 = nn.Linear(in_features=512, out_features=128)
        self.afc1 = nn.Linear(in_features=128, out_features=128)
        self.afc2 = nn.Linear(in_features=128, out_features=128)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)  # output bn*32*120
        # self.mp = nn.MaxPool1d(kernel_size=4)
        self.dp = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1)  # output bn*32*113
        self.fc3 = nn.Linear(in_features=32 * 113, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=2)
        self.fc5 = nn.Linear(in_features=1024, out_features=2)
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

        # afeat = self.afc1(afeat)
        # afeat = F.relu(afeat)
        # afeat = self.afc2(afeat)
        # afeat = F.relu(afeat)

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

        result = F.softmax(vafeat)

        return result


class VA_LSTM(nn.Module):
    def __init__(self, num_layer=5):
        super(VA_LSTM, self).__init__()

        self.num_layer = num_layer
        self.v_lstm = nn.LSTM(input_size=1024, hidden_size=128, num_layers=num_layer, batch_first=True,
                              bidirectional=True)
        self.a_lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=num_layer, batch_first=True,
                              bidirectional=True)

        self.fc1 = nn.Linear(128 * 2, 128 * 2)
        self.fc2 = nn.Linear(128 * 2, 128 * 2)
        self.fc3 = nn.Linear(128 * 2, 1)

    def forward(self, vfeat, afeat):
        bz = vfeat.size(0)

        h_0_v = Variable(torch.Tensor(self.num_layer * 2, bz, 128)).cuda()
        c_0_v = Variable(torch.Tensor(self.num_layer * 2, bz, 128)).cuda()
        nn.init.orthogonal(h_0_v)
        nn.init.orthogonal(c_0_v)
        vfeat = self.v_lstm(vfeat, (h_0_v, c_0_v))[0]
        vfeat = vfeat[:, 119, :]

        h_0_a = Variable(torch.Tensor(self.num_layer * 2, bz, 128)).cuda()
        c_0_a = Variable(torch.Tensor(self.num_layer * 2, bz, 128)).cuda()
        nn.init.orthogonal(h_0_a)
        nn.init.orthogonal(c_0_a)
        afeat = self.a_lstm(afeat, (h_0_a, c_0_a))[0]
        afeat = afeat[:, 119, :]

        vfeat = F.relu(self.fc1(vfeat))
        vfeat = F.relu(self.fc2(vfeat))
        vfeat = F.relu(self.fc3(vfeat))

        afeat = F.relu(self.fc1(afeat))
        afeat = F.relu(self.fc2(afeat))
        afeat = F.relu(self.fc3(afeat))

        vafeat = vfeat - afeat
        dis = torch.norm(vafeat, p=2, dim=1)

        return dis


class metric_loss(nn.Module):
    def __init__(self):
        super(metric_loss, self).__init__()

    def forward(self, dis, target, margin=1):
        bs = dis.size(0)

        loss_posi = F.relu(torch.mean((1 - target) * dis))
        loss_nega = torch.mean(F.relu(margin - target * dis))
        loss_balance = F.relu(0.5 - (torch.mean(dis[0:bs / 2 - 1]) - torch.mean(dis[bs / 2:bs - 1])))
        print list(loss_posi.data)[0], list(loss_nega.data)[0], list(loss_balance.data)[0]

        return loss_posi + loss_nega + loss_balance


class conv_loss_dqy(torch.nn.Module):
    def __init__(self):
        super(conv_loss_dqy, self).__init__()

    def forward(self, sim, label):
        length = len(sim[:, 1])
        loss1 = torch.mean(torch.pow((1 - label) * sim[:, 1], 2))
        loss2 = torch.mean(torch.pow(label * sim[:, 0], 2))
        # loss3 = 2.2 - (torch.mean(sim[0:length / 2 - 1]) - torch.mean(sim[length / 2:length - 1]))
        loss3 = 0.9 - torch.mean(torch.abs(sim[:, 0] - sim[:, 1]))
        return 1 * loss1 + 1 * loss2 + 0 * loss3


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


class Topk_loss(torch.nn.Module):
    def __init__(self):
        super(Topk_loss, self).__init__()

    def forward(self, sim_0, sim_1, k=5):
        bn = sim_0.size()[0]

        loss3 = 1 - torch.mean(torch.diag(sim_0) - torch.diag(sim_1))
        loss4 = 1 - torch.mean((sim_1 - torch.diag(torch.diag(sim_1))) - (sim_0 - torch.diag(torch.diag(sim_0))))
        sort, indices = torch.sort(sim_0, dim=1, descending=True)
        np_indices = indices.cpu().data.numpy()
        topk = np_indices[:, 0:k - 1]
        wrong = 0
        for i in range(bn):
            if i not in topk[i, :]:
                wrong += 1
                try:
                    loss1 += sim_1[i, i]
                except Exception:
                    loss1 = sim_1[i, i]
        loss1 = loss1 / wrong
        sim_0 = sim_0 - torch.diag(torch.diag(sim_0))
        loss2 = torch.mean(torch.max(sim_0, dim=1)[0])

        # print list(loss1.data)[0], list(loss2.data)[0]
        print list(loss3.data)[0], list(loss4.data)[0]
        return 0 * loss1 + 0 * loss2 + 1 * loss3 + 1 * loss4
