import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

USE_CUDA = True


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


class VA_lstm(nn.Module):
    def __init__(self, hidden_size=128, num_layers=5):
        super(VA_lstm, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection = False
        self.num_direction = 1
        if self.bidirection:
            self.num_direction = 2
        self.valstm_hidden_size = self.hidden_size * 2 * self.num_direction

        self.vlstm = nn.LSTM(input_size=1024, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
                             batch_first=True, bidirectional=self.bidirection)

        self.alstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
                             batch_first=True, bidirectional=self.bidirection)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)  # output bn * 32 * 120
        self.dp = nn.Dropout(p=0.5)
        self.vfc = nn.Linear(1024, 128)
        self.fc1 = nn.Linear(120 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.vafc1 = nn.Linear(256, 256)
        self.vafc2 = nn.Linear(256, 256)
        self.vafc3 = nn.Linear(256, 2)

        self.Linear_init()

    def forward(self, vfeat, afeat):
        bs = vfeat.size(0)

        vfeat = self.vlstm(vfeat, self.param_init(batch_size=bs))[0]
        afeat = self.alstm(afeat, self.param_init(batch_size=bs))[0]

        vfeat = vfeat[:, 119, :]
        afeat = afeat[:, 119, :]
        va = torch.cat((vfeat, afeat), dim=1)
        va = F.relu(self.vafc1(va))
        va = F.relu(self.vafc2(va))
        sim = F.softmax(self.vafc3(va))

        #
        # vfeat = F.relu(self.vfc(vfeat))
        # vfeat = vfeat.resize(bs, 1, 1, 120 * 128)
        # afeat = afeat.resize(bs, 1, 1, 120 * 128)
        # vafeat = torch.cat((vfeat, afeat), dim=2)
        # vafeat = self.conv1(vafeat)
        # vafeat = self.dp(vafeat)

        # vafeat = vafeat.view(bs, -1)
        # vafeat = F.relu(self.fc1(vafeat))
        # vafeat = F.relu(self.fc2(vafeat))
        # sim = F.softmax(self.fc3(vafeat))

        return sim

    def param_init(self, batch_size, hidden_size=None):
        if hidden_size is None:
            hidden_size = self.hidden_size
        bs = batch_size
        h_0 = Variable(torch.zeros(self.num_layers * self.num_direction, bs, hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers * self.num_direction, bs, hidden_size))
        torch.nn.init.xavier_normal(h_0)
        torch.nn.init.xavier_normal(c_0)
        if USE_CUDA:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return h_0, c_0

    def Linear_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)


class lstm_loss(nn.Module):
    def __init__(self):
        super(lstm_loss, self).__init__()

    def forward(self, sim, target, margin=1):
        sim_0 = sim[:, 0]
        sim_1 = sim[:, 1]
        loss_posi = torch.mean(F.relu((1 - target) * sim_1))
        loss_nega = torch.mean(F.relu(target * sim_0))
        loss = (loss_nega + loss_posi) / 2

        print(loss_posi.data[0], loss_nega.data[0])
        return loss
