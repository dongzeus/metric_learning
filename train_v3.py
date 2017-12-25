#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models_v3 as models
import evaluate_v3 as evaluate
from dataset import VideoFeatDataset
from tools.config_tools import Config
from tools import utils

import matplotlib as mpl

mpl.use('Agg')

from matplotlib import pyplot as plt

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))

train_dataset = VideoFeatDataset(opt.data_dir, flist=opt.flist)

print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    torch.manual_seed(opt.manualSeed)
else:
    if int(opt.ngpu) == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
            torch.cuda.manual_seed(opt.manualSeed)
            cudnn.benchmark = True
print('Random Seed: {0}'.format(opt.manualSeed))


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt,test_video_loader, test_audio_loader, opts_test):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()

    global dis1_rec
    global dis2_rec
    global loss_rec

    for i, (vfeat, afeat) in enumerate(train_loader):
        #shuffling the index orders
        # bz = vfeat.size()[0]
        # orders = np.arange(bz).astype('int32')
        # shuffle_orders = orders.copy()
        # np.random.shuffle(shuffle_orders)
        #
        # # creating a new data with the shuffled indices
        # afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()
        #
        # # concat the vfeat and afeat respectively
        # afeat0 = torch.cat((afeat, afeat2), 0)
        # vfeat0 = torch.cat((vfeat, vfeat), 0)
        #
        # # generating the labels
        # # 1. the labels for the shuffled feats
        # label1 = (orders == shuffle_orders + 0).astype('float32')
        # target1 = torch.from_numpy(label1)
        #
        # # 2. the labels for the original feats
        # label2 = label1.copy()
        # label2[:] = 1
        # target2 = torch.from_numpy(label2)
        #
        # # concat the labels together
        # target = torch.cat((target2, target1), 0)
        # target = 1 - target
        #
        # # transpose the feats
        # # vfeat0 = vfeat0.transpose(2, 1)
        # # afeat0 = afeat0.transpose(2, 1)
        #
        #
        # # put the data into Variable
        # vfeat_var = Variable(vfeat0)
        # afeat_var = Variable(afeat0)
        # target_var = Variable(target)
        #
        # # if you have gpu, then shift data to GPU
        # if opt.cuda:
        #     vfeat_var = vfeat_var.cuda()
        #     afeat_var = afeat_var.cuda()
        #     target_var = target_var.cuda()
        # sim,dis1,dis2 = model(vfeat_var,afeat_var)
        ##### for N pair loss
        vfeat = Variable(vfeat)
        afeat = Variable(afeat)
        if opt.cuda:
            vfeat = vfeat.cuda()
            afeat = afeat.cuda()
        bz = vfeat.size()[0]
        for k in np.arange(bz):
            cur_vfeat = vfeat[k].clone()
            vfeat_k = cur_vfeat.repeat(bz, 1, 1)
            sim_k, dis1, dis2 = model(vfeat_k, afeat)
            sim_k_0 = sim_k[:,0]
            sim_k_1 = sim_k[:,1]
            sim_k_0 = sim_k_0.resize(1,bz)
            sim_k_1 = sim_k_1.resize(1,bz)
            if k == 0:
                sim_0 = sim_k_0.clone()
                sim_1 = sim_k_1.clone()
            else:
                sim_0 = torch.cat((sim_0, sim_k_0), dim=0)
                sim_1 = torch.cat((sim_1, sim_k_1), dim=0)
        loss = criterion(sim_0,sim_1)


        # record the loss and distance to plot later
        # dis1_rec.append(list(dis1.data)[0])
        # dis2_rec.append(list(dis2.data)[0])
        loss_rec.append(list(loss.data)[0])

        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        # utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)
    evaluate.test(test_video_loader, test_audio_loader, model, opts_test)



def main():
    global opt
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers))

    # create model
    model = models.VAMetric_conv()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    # criterion = models.conv_loss_dqy()
    criterion = models.N_pair_loss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr,
    #                      momentum=opt.momentum,
    #                      weight_decay=opt.weight_decay)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    # optimizer = optim.Adadelta(params=model.parameters(), lr=opt.lr)
    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)  # poly policy
    scheduler = LR_Policy(optimizer, lambda_lr)

    resume_epoch = 0

    global dis1_rec
    global dis2_rec
    global loss_rec

    loss_rec = []
    dis1_rec = []
    dis2_rec = []

    ######### to test each epoch
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="evaluation configuration",
                      default="./configs/test_config.yaml")

    (opts_test, args) = parser.parse_args()
    opts_test = Config(opts_test.config)
    test_video_dataset = VideoFeatDataset(opts_test.data_dir, opts_test.video_flist, which_feat='vfeat')
    test_audio_dataset = VideoFeatDataset(opts_test.data_dir, opts_test.audio_flist, which_feat='afeat')
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opts_test.batchSize,
                                                    shuffle=False, num_workers=int(opts_test.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opts_test.batchSize,
                                                    shuffle=False, num_workers=int(opts_test.workers))

    ########

# another test for git
    for epoch in range(resume_epoch, opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt,test_video_loader, test_audio_loader, opts_test)
        scheduler.step()

        ##################################
        # save checkpoints
        ##################################

        # save model every 10 epochs
        if ((epoch + 1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(opt.checkpoint_folder, opt.prefix, epoch + 1)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(loss_rec)
    plt.legend('loss')
    plt.subplot(1, 2, 2)
    plt.plot(dis1_rec)
    plt.plot(dis2_rec)
    plt.legend(('distance between positives', 'distance between negatives'))
    plt.show()
    plt.savefig("./figures/conv.jpg")


if __name__ == '__main__':
    main()
