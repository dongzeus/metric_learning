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

import models_ensem as models
from dataset_ensem import VideoFeatDataset
from tools.config_tools import Config
from tools import utils
from sklearn.decomposition import PCA
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

tds_ls = []
for i in range(opt.model_number):
    if i == 0:
        tds_ls.append(VideoFeatDataset(root=opt.data_dir, flist=opt.flist, test_list=opt.test_flist,
                                       test_number=opt.test_number, bagging=True, creat_test=True))
    else:
        tds_ls.append(VideoFeatDataset(root=opt.data_dir, flist=opt.flist, bagging=True, creat_test=False,
                                       test_list_pass=tds_ls[0].get_ori_pathlist()))

# =================== creat test set before import evaluate ===================
import evaluate_ensem as evaluate

# =================== creat test set before import evaluate ===================



print('number of train samples is: {0}'.format(len(tds_ls[0])))
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


def pca_tensor(tensor, dim, feat, pr=False):
    if feat == 'vfeat':
        n_component = opt.vfeat_pca
    elif feat == 'afeat':
        n_component = opt.afeat_pca
    else:
        exit()

    bs = tensor.size(0)
    pca = PCA(n_components=n_component)
    if isinstance(tensor, Variable):
        tensor_np = tensor.data.numpy()
    else:
        tensor_np = tensor.numpy()
    tensor_np.resize(bs * 120, dim)
    pca.fit(tensor_np)
    tensor_np = pca.transform(tensor_np)
    tensor_np.resize(bs, 120, n_component)
    tensor_new = torch.from_numpy(tensor_np)
    if isinstance(tensor, Variable):
        tensor_new = Variable(tensor_new)
        if opt.cuda:
            tensor_new = tensor_new.cuda()

    if pr:
        print('afeat PCA variance ratio:')
        print(pca.explained_variance_ratio_)
        print(np.sum(pca.explained_variance_ratio_))
    return tensor_new


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt, num):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()

    global positive_rec
    global negative_rec
    global loss_rec

    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders

        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # concat the vfeat and afeat respectively
        afeat0 = torch.cat((afeat, afeat2), 0)
        vfeat0 = torch.cat((vfeat, vfeat), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (orders == shuffle_orders + 0).astype('float32')
        target1 = torch.from_numpy(label1)

        # 2. the labels for the original feats
        label2 = label1.copy()
        label2[:] = 1
        target2 = torch.from_numpy(label2)

        # concat the labels together
        target = torch.cat((target2, target1), 0)
        target = 1 - target

        # transpose the feats
        # vfeat0 = vfeat0.transpose(2, 1)
        # afeat0 = afeat0.transpose(2, 1)


        # put the data into Variable
        vfeat_var = Variable(vfeat0)
        afeat_var = Variable(afeat0)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()
        sim = model(vfeat_var, afeat_var)
        loss = criterion(sim, target_var)

        loss_rec.append(list(loss.data)[0])
        positive_rec.append(list(torch.mean(sim[0:bz - 1]).data)[0])
        negative_rec.append(list(torch.mean(sim[bz:bz * 2 - 1]).data)[0])

        # ##### for N pair loss
        # vfeat = Variable(vfeat)
        # afeat = Variable(afeat)
        # if opt.cuda:
        #     vfeat = vfeat.cuda()
        #     afeat = afeat.cuda()
        # bz = vfeat.size()[0]
        # for k in np.arange(bz):
        #     cur_vfeat = vfeat[k].clone()
        #     vfeat_k = cur_vfeat.repeat(bz, 1, 1)
        #     sim_k = model(vfeat_k, afeat)
        #     sim_k_0 = sim_k[:, 0]
        #     sim_k_1 = sim_k[:, 1]
        #     sim_k_0 = sim_k_0.resize(1, bz)
        #     sim_k_1 = sim_k_1.resize(1, bz)
        #     if k == 0:
        #         sim_0 = sim_k_0.clone()
        #         sim_1 = sim_k_1.clone()
        #     else:
        #         sim_0 = torch.cat((sim_0, sim_k_0), dim=0)
        #         sim_1 = torch.cat((sim_1, sim_k_1), dim=0)
        # loss = criterion(sim_0, sim_1)
        #
        # loss_rec.append(list(loss.data)[0])
        # positive_rec.append(list(torch.mean(torch.diag(sim_0)).data)[0])
        # sim_0 = sim_0 - torch.diag(torch.diag(sim_0))
        # negative_rec.append(list(torch.mean(sim_0).data)[0])


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
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'No.{} Epoch: [{}][{}/{}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                num, epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)


def main():
    global opt
    # train data loader
    tl_ls = []
    for tds in tds_ls:
        tl_ls.append(torch.utils.data.DataLoader(tds, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers)))

    # create model
    model_ls = []
    for i in range(opt.model_number):
        m = models.VA_lstm()
        # m = models.VAMetric_conv()
        model_ls.append(m)

    if opt.init_model_epoch != '':

        for i in range(opt.model_number):
            path = '{0}/{1}_state_epoch{2}_model{3}.pth'.format(opt.checkpoint_folder, opt.prefix,
                                                                opt.init_model_epoch, i + 1)
            print('loading pretrained model from {0}'.format(path))
            model_ls[i].load_state_dict(torch.load(path))

    # Contrastive Loss
    # criterion = models.conv_loss_dqy()
    # criterion = models.N_pair_loss()
    # criterion = models.Topk_loss()
    criterion = models.lstm_loss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        for i in range(opt.model_number):
            model_ls[i] = model_ls[i].cuda()
        criterion = criterion.cuda()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr,
    #                      momentum=opt.momentum,
    #                      weight_decay=opt.weight_decay)

    opt_ls = []
    for m in model_ls:
        op = optim.Adam(m.parameters(), lr=opt.lr)
        # op = optim.SGD(m.parameters(), lr=opt.lr,
        #                momentum=opt.momentum,
        #                weight_decay=opt.weight_decay)
        opt_ls.append(op)

    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    # optimizer = optim.Adadelta(params=model.parameters(), lr=opt.lr)
    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)  # poly policy
    scheduler_ls = []
    for op in opt_ls:
        scheduler_ls.append(LR_Policy(op, lambda_lr))

    resume_epoch = 0

    global positive_rec
    global negative_rec
    global loss_rec

    loss_rec = []
    positive_rec = []
    negative_rec = []

    ######### to test each epoch
    parser = OptionParser()
    parser.add_option('--config',
                      type=str,
                      help="evaluation configuration",
                      default="./configs/test_config.yaml")

    (opts_test, args) = parser.parse_args()
    opts_test = Config(opts_test.config)
    test_video_dataset = VideoFeatDataset(root=opts_test.data_dir, flist=opts_test.video_flist, which_feat='vfeat',
                                          creat_test=0)
    test_audio_dataset = VideoFeatDataset(root=opts_test.data_dir, flist=opts_test.audio_flist, which_feat='afeat',
                                          creat_test=0)
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
        for i in range(opt.model_number):
            train(train_loader=tl_ls[i], model=model_ls[i], criterion=criterion, optimizer=opt_ls[i], epoch=epoch + 1,
                  opt=opt, num=i + 1)
            scheduler_ls[i].step()
        ##################################
        # save checkpoints
        ##################################

        # save model every 10 epochs
        if ((epoch + 1) % opt.epoch_save) == 0:
            for i in range(opt.model_number):
                path_checkpoint = '{0}/{1}_state_epoch{2}_model{3}.pth'.format(opt.checkpoint_folder, opt.prefix,
                                                                               epoch + 1, i + 1)
                utils.save_checkpoint(model_ls[i].state_dict(), path_checkpoint)

        if ((epoch + 1) % opt.epoch_plot) == 0:
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.plot(loss_rec)
            plt.legend('loss')
            plt.subplot(1, 2, 2)
            plt.plot(positive_rec)
            plt.plot(negative_rec)
            plt.legend(('simmilarity of positives', 'simmilarity of negatives'))
            plt.show()
            plt.savefig('./figures/result{0}.jpg'.format(epoch + 1))
            plt.close()
        if ((epoch + 1) % opt.epoch_test) == 0:
            evaluate.test(test_video_loader, test_audio_loader, model_ls, opts_test)


if __name__ == '__main__':
    main()
