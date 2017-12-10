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
                stream = h_t2.view(h_t2.size(0),1,-1)
            else:
                stream = torch.cat((stream,h_t2.view(h_t2.size(0),1,-1)),dim=1)
        # aggregated feature

        return stream



class VAMetric_conv(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric_conv, self).__init__()

        self.vLSTM = FeatLSTM(1024,512,128)
        self.aLSTM = FeatLSTM(128,128,128)

        self.vfc1 = nn.Linear(in_features=1024, out_features=512)
        self.vfc2 = nn.Linear(in_features=512, out_features=128)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)
        # self.mp = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1)
        self.fc3 = nn.Linear(in_features=32 * 113, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=2)
        self.fc5 = nn.Linear(in_features=512, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=2)
        self.init_params()

    def init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):

        #vfeat = self.vLSTM(vfeat)
        #afeat = self.aLSTM(afeat)

        #vfeat = self.vfc1(vfeat)
        #vfeat = F.relu(vfeat)
        #vfeat = self.vfc2(vfeat)
        #vfeat = F.relu(vfeat)

        vfeat = vfeat.view(vfeat.size(0), 1, 1, -1)
        afeat = afeat.view(afeat.size(0), 1, 1, -1)
        afeat = afeat.repeat(1,1,1,8)

        vafeat = torch.cat((vfeat, afeat), dim=2)
        vafeat = self.conv1(vafeat)
        vafeat = vafeat.view(vafeat.size(0), vafeat.size(1), -1)
        vafeat = self.conv2(vafeat)

        vafeat = vafeat.view([vafeat.size(0), -1])
        vafeat = self.fc3(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc4(vafeat)
       # vafeat = F.relu(vafeat)
       # vafeat = self.fc5(vafeat)
        #vafeat = F.relu(vafeat)
        #vafeat = self.fc6(vafeat)


        result = F.softmax(vafeat)

        # vafeat = F.relu(vafeat)
        # vafeat = self.fc5(vafeat)
        # vafeat = F.relu(vafeat)
        # vafeat = self.fc6(vafeat)

        # vafeat = 1.2 * F.tanh(vafeat)

        return result, torch.mean(result[0:result.size(0) / 2 - 1], 0), torch.mean(
            result[result.size(0) / 2:vafeat.size(0) - 1], 0)


# only to test the git hub

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
