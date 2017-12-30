import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from optparse import OptionParser
from tools.config_tools import Config

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
USE_CUDA = opt.cuda


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


# class VA_lstm(nn.Module):
#     def __init__(self, hidden_size=128 * 3, num_layers=2):
#         super(VA_lstm, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirection = True
#         self.num_direction = 1
#         if self.bidirection:
#             self.num_direction = 2
#
#         self.vlstm = nn.LSTM(input_size=1024 * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
#                              batch_first=True, bidirectional=self.bidirection)
#
#         self.alstm = nn.LSTM(input_size=128 * 3, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.1,
#                              batch_first=True, bidirectional=self.bidirection)
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 128 * 3 * 2),
#                                stride=128 * 3 * 2)  # output bn * 16 * 118
#         self.mp1 = nn.MaxPool1d(kernel_size=3)  # bn * 16 * 39
#         self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, stride=1)  # bn * 16 * 36
#
#         self.dp = nn.Dropout(p=0.3)
#         self.vafc1 = nn.Linear(8 * 36, 256)
#         self.vafc2 = nn.Linear(256, 2)
#         self.vafc3 = nn.Linear(1024, 2)
#
#         self.Linear_init()
#
#     def forward(self, vfeat, afeat):
#         bs = vfeat.size(0)
#
#         for seq in range(118):
#             if seq == 0:
#                 vfeat_3 = vfeat[:, seq:seq + 3, :].resize(bs, 1, 3 * 1024)
#                 afeat_3 = afeat[:, seq:seq + 3, :].resize(bs, 1, 3 * 128)
#
#             else:
#                 vfeat_3 = torch.cat((vfeat_3, vfeat[:, seq:seq + 3, :].resize(bs, 1, 3 * 1024)), dim=1)
#                 afeat_3 = torch.cat((afeat_3, afeat[:, seq:seq + 3, :].resize(bs, 1, 3 * 128)), dim=1)
#
#         vlstm = self.vlstm(vfeat_3, self.param_init(batch_size=bs, hidden_size=self.hidden_size))[0]
#         alstm = self.alstm(afeat_3, self.param_init(batch_size=bs, hidden_size=self.hidden_size))[0]
#
#         vlstm = vlstm.resize(bs, 1, 1, 118 * self.hidden_size * 2)
#         alstm = alstm.resize(bs, 1, 1, 118 * self.hidden_size * 2)
#
#         va = torch.cat((vlstm, alstm), dim=2)
#         va = self.conv1(va)
#         va = va.view(va.size(0), va.size(1), -1)
#         va = self.dp(va)
#         va = self.mp1(va)
#         va = self.conv2(va)
#         va = va.view(bs, -1)
#
#         va = F.relu(self.vafc1(va))
#         sim = F.softmax(self.vafc2(va))
#
#         return sim
#
#     def param_init(self, batch_size, hidden_size=None):
#         if hidden_size is None:
#             hidden_size = self.hidden_size
#         bs = batch_size
#         h_0 = Variable(torch.zeros(self.num_layers * self.num_direction, bs, hidden_size))
#         c_0 = Variable(torch.zeros(self.num_layers * self.num_direction, bs, hidden_size))
#         torch.nn.init.xavier_normal(h_0)
#         torch.nn.init.xavier_normal(c_0)
#         if USE_CUDA:
#             h_0 = h_0.cuda()
#             c_0 = c_0.cuda()
#
#         return h_0, c_0
#
#     def Linear_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal(m.weight)
#                 nn.init.constant(m.bias, 0)

class VA_lstm(nn.Module):
    def __init__(self, hidden_size=128 * 3, num_layers=2):
        super(VA_lstm, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection = True
        self.num_direction = 1
        if self.bidirection:
            self.num_direction = 2

        self.vfc1 = nn.Linear(1024,128)
        self.afc1 = nn.Linear(128,128)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128*5),
                               stride=128)  # output bn * 16 * 118

        self.dp = nn.Dropout(p=0.31)
        self.vafc1 = nn.Linear(32*116, 1024)
        self.vafc2 = nn.Linear(1024, 2)
        self.Linear_init()

    def forward(self, vfeat, afeat):
        bs = vfeat.size(0)

        vfeat = F.relu(self.vfc1(vfeat))
        afeat = F.relu((self.afc1(afeat)))

        vlstm = vfeat.resize(bs, 1, 1, 120*128)
        alstm = afeat.resize(bs, 1, 1, 120*128)

        va = torch.cat((vlstm, alstm), dim=2)
        va = self.conv1(va)
        va = va.view(va.size(0), va.size(1), -1)
        va = self.dp(va)

        va = va.view(bs, -1)

        va = F.relu(self.vafc1(va))
        sim = F.softmax(self.vafc2(va))

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
        bs = sim.size(0)
        sim_0 = sim[:, 0]
        sim_1 = sim[:, 1]
        loss_posi = torch.mean(F.relu(torch.pow(sim_1[0:bs / 2], 1)))
        loss_nega = torch.mean(F.relu(torch.pow(sim_0[bs / 2:bs], 1)))

        loss_balance1 = F.relu(
            0.9 - (torch.mean(torch.pow(sim_0[0:bs / 2], 1)) - torch.mean(torch.pow(sim_0[bs / 2:bs], 1))))
        loss_balance2 = F.relu(
            0.9 - (torch.mean(torch.pow(sim_1[bs / 2:bs], 1)) - torch.mean(torch.pow(sim_1[0:bs / 2], 1))))

        loss = 0.1 * loss_nega + 0.1 * loss_posi + 1 * loss_balance1 + 0 * loss_balance2

        # print(loss_posi.data[0], loss_nega.data[0], loss_balance1.data[0], loss_balance2.data[0])
        return loss

        # class lstm_loss(nn.Module):
        #     def __init__(self):
        #         super(lstm_loss, self).__init__()
        #
        #     def forward(self, sim, target, margin=1):
        #         bs = sim.size(0)
        #         sim_0 = sim[:, 0]
        #         sim_1 = sim[:, 1]
        #         loss_posi = torch.mean(F.relu(torch.pow(sim_1[0:bs / 2], 1)))
        #         loss_nega = torch.mean(F.relu(torch.pow(sim_0[bs / 2:bs], 1)))
        #
        #         loss_balance1 = torch.mean(torch.clamp(margin - (sim_0[0:bs / 2] - sim_0[bs / 2:bs]), min=0))
        #
        #         loss_balance2 = F.relu(
        #             0.9 - (torch.mean(torch.pow(sim_1[bs / 2:bs], 1)) - torch.mean(torch.pow(sim_1[0:bs / 2], 1))))
        #
        #         loss = 0.1 * loss_nega + 0.1 * loss_posi + 1 * loss_balance1 + 0 * loss_balance2
        #
        #         print(loss_posi.data[0], loss_nega.data[0], loss_balance1.data[0])
        #         return loss
        #


        # def forward(self, sim, target, margin=0.5):
        #     bs = sim.size(0)
        #     sim_0 = sim[:, 0]
        #     sim_1 = sim[:, 1]
        #     posi_0 = sim_0[0:bs / 2]
        #     posi_1 = sim_1[0:bs / 2]
        #     nega_0 = sim_0[bs / 2:bs]
        #     nega_1 = sim_1[bs / 2:bs]
        #
        #     loss_posi = torch.mean(F.relu(torch.pow(sim_1[0:bs / 2], 1)))
        #     loss_nega = torch.mean(F.relu(torch.pow(sim_0[bs / 2:bs], 1)))
        #
        #     loss_balance1 = torch.mean(torch.pow(torch.clamp(margin - (posi_0 - nega_0), min=0), 1))
        #     loss_balance2 = torch.mean(torch.pow(torch.clamp(margin - (nega_1 - posi_1), min=0), 1))
        #
        #     loss = 1 * loss_balance1 + 1 * loss_balance2
        #
        #     print(loss_balance1.data[0], loss_balance2.data[0])
        #     return loss
