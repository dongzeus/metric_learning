import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


# test for git

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, out_size=128):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size, )
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
        # aggregated feature
        feat = h_t2
        return feat


# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 512, 128)
        self.AFeatPool = FeatAggregate(128, 128, 128)
        self.fc = nn.Linear(128, 64)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = self.fc(vfeat)
        afeat = self.fc(afeat)

        distance = F.pairwise_distance(vfeat, afeat)

        return distance, torch.mean(distance[0:vfeat.size(0) / 2 - 1]), torch.mean(
            distance[vfeat.size(0) / 2:vfeat.size(0) - 1])


# Visual-audio multimodal metric learning: MaxPool+FC
class VAMetric2(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric2, self).__init__()
        self.mp = nn.MaxPool1d(framenum)
        self.vfc = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 96)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):
        # aggregate the visual features
        vfeat = self.mp(vfeat)
        vfeat = vfeat.view(-1, 1024)
        vfeat = F.relu(self.vfc(vfeat))
        vfeat = self.fc(vfeat)

        # aggregate the auditory features
        afeat = self.mp(afeat)
        afeat = afeat.view(-1, 128)
        afeat = self.fc(afeat)

        return F.pairwise_distance(vfeat, afeat)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        length = len(dist)
        loss = torch.mean((1 - label) * torch.pow(dist, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        loss2 = 1 - torch.mean(torch.abs(dist[0:length / 2 - 1] - dist[length / 2:length - 1]))
        return loss + loss2


class VAMetric_conv(nn.Module):
    def __init__(self, framenum=120):
        super(VAMetric_conv, self).__init__()
        self.vfc1 = nn.Linear(in_features=1024, out_features=512)
        self.vfc2 = nn.Linear(in_features=512, out_features=128)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 128), stride=128)
        # self.mp = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1)
        self.fc3 = nn.Linear(in_features=32 * 113, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, vfeat, afeat):

        vfeat = self.vfc1(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.vfc2(vfeat)
        vfeat = F.relu(vfeat)

        # vfeat = self.bn(vfeat)
        # afeat = self.bn(afeat)

        vfeat = vfeat.view(vfeat.size(0), 1, 1, -1)
        afeat = afeat.view(afeat.size(0), 1, 1, -1)

        vafeat = torch.cat((vfeat, afeat), dim=2)
        vafeat = self.conv1(vafeat)
        vafeat = vafeat.view(vafeat.size(0), vafeat.size(1), -1)
        vafeat = self.conv2(vafeat)
        # vafeat = self.mp(vafeat)

        vafeat = vafeat.view([vafeat.size(0), -1])
        vafeat = self.fc3(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc4(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc5(vafeat)
        vafeat = F.relu(vafeat)
        vafeat = self.fc6(vafeat)

        return vafeat




        # vafeat = F.relu(vafeat)
        # vafeat = self.fc5(vafeat)
        # vafeat = F.relu(vafeat)
        # vafeat = self.fc6(vafeat)

        # vafeat = 1.2 * F.tanh(vafeat)

        # return result, torch.mean(result[0:result.size(0) / 2 - 1], 0), torch.mean(
        #   result[result.size(0) / 2:vafeat.size(0) - 1], 0)


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


class N_pair_loss(torch.nn.Module):
    def __init__(self):
        super(N_pair_loss, self).__init__()

    def forward(self, dis, u=0.1, margin=1):
        # u is the parameter for regularization loss constant

        bn = dis.size()[0]

        loss = 0
        for i in range(bn):

            Dij = dis[i, i]

            Dik = dis[i, :].clone()
            Dik[i] = 0
            Djk = dis[:, i].clone()
            Djk[i] = 0

            loss_i = torch.log(
                torch.sum(torch.exp(margin * torch.autograd.Variable(torch.ones(Dik.size())).cuda() - Dik) + torch.exp(
                    margin * torch.autograd.Variable(torch.ones(Djk.size())).cuda() - Djk), dim=0)) + Dij
            if torch.norm(loss_i, p=1).data[0] < 0:
                continue
            else:
                loss_i = torch.pow(loss_i, 2)
                loss = loss + loss_i
        loss = loss / (2 * bn)

        # S = torch.mm(vfeat, torch.t(afeat))
        # S = torch.exp(S)
        # S_sum_1 = torch.sum(S, 0)
        # S_sum_2 = torch.sum(S, 1)
        # diag = torch.diag(S)
        #
        # loss_v = (-1) * torch.sum(torch.log(torch.div(diag, S_sum_1))) / bn
        # loss_a = (-1) * torch.sum(torch.log(torch.div(diag, S_sum_2))) / bn


        return loss
