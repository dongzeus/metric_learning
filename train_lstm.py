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

import models_lstm as models
from dataset_lstm import VideoFeatDataset
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

tds_ls = []
#######################################################
# change the first bagging to True when ensemble many models
#######################################################
for i in range(opt.model_number):
    if i == 0:
        tds_ls.append(VideoFeatDataset(root=opt.data_dir, flist=opt.flist, test_list=opt.test_flist,
                                       test_number=opt.test_number, bagging=False, creat_test=True))
    else:
        tds_ls.append(VideoFeatDataset(root=opt.data_dir, flist=opt.flist, bagging=True, creat_test=False,
                                       test_list_pass=tds_ls[0].get_ori_pathlist()))

# =================== creat test set before import evaluate ===================
import evaluate_lstm as evaluate

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


# training function for metric learning
def train(train_loader, encoder, decoder, criterion, encoder_optim, decoder_optim, epoch, opt, num):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    encoder.train()
    decoder.train()

    end = time.time()

    global sim_rec
    global loss_rec

    teaching_ratio = 0.5

    for i, (vfeat, afeat) in enumerate(train_loader):

        bs = vfeat.size()[0]
        seq_length = 120
        vfeat = Variable(vfeat)
        target = Variable(afeat)

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        loss = 0

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat = vfeat.cuda()
            target = target.cuda()

        # use video features to generate encoder_output and encoder_hidden (to be the initial hidden for decoder)
        encoder_hidden = encoder.init_hidden(batch_size=bs)
        encoder_output, encoder_hidden = encoder(vfeat, encoder_hidden)

        # decoder
        decoder_hidden = encoder_hidden
        decoder_input = encoder_output[:, 119, :]  # bs * 128
        decoder_context = torch.mean(encoder_output, dim=1)  # bs * 128

        teaching = random.random() < teaching_ratio

        if teaching:
            for seq in range(seq_length):
                audio_output, decoder_context, decoder_hidden, attn_weights = decoder(decoder_input, decoder_context,
                                                                                      decoder_hidden, encoder_output)
                loss += criterion(audio_output, target[:, seq, :])
                decoder_input = target[:, seq, :]
        else:
            for seq in range(seq_length):
                audio_output, decoder_context, decoder_hidden, attn_weights = decoder(decoder_input, decoder_context,
                                                                                      decoder_hidden, encoder_output)
                loss += criterion(audio_output, target[:, seq, :])
                decoder_input = audio_output

        loss = loss / seq_length
        loss_rec.append(loss.data[0])

        losses.update(loss.data[0], vfeat.size(0))
        loss.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(), opt.gradient_clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), opt.gradient_clip)

        encoder_optim.step()
        decoder_optim.step()

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
        encoder = models.Encoder()
        decoder = models.AttnDecoder()
        model_ls.append([encoder, decoder])

    # if opt.init_model_epoch != '':
    #     for i in range(opt.model_number):
    #         path = '{0}/{1}_state_epoch{2}_model{3}.pth'.format(opt.checkpoint_folder, opt.prefix,
    #                                                             opt.init_model_epoch, i + 1)
    #         print('loading pretrained model from {0}'.format(path))
    #         model_ls[i].load_state_dict(torch.load(path))

    criterion = models.pairwise_loss()

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        for i in range(opt.model_number):
            cp = model_ls[i]
            cp[0] = cp[0].cuda()
            cp[1] = cp[1].cuda()
        criterion = criterion.cuda()

    opt_ls = []
    for m in model_ls:
        encoder = m[0]
        decoder = m[1]
        encoder_optim = optim.Adam(encoder.parameters(), lr=opt.lr)
        decoder_optim = optim.Adam(decoder.parameters(), lr=opt.lr)
        op = [encoder_optim, decoder_optim]
        opt_ls.append(op)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)  # poly policy
    scheduler_ls = []
    for op in opt_ls:
        en = LR_Policy(op[0], lambda_lr)
        de = LR_Policy(op[1], lambda_lr)
        scheduler_ls.append([en, de])

    resume_epoch = 0

    global positive_rec
    global negative_rec
    global loss_rec

    loss_rec = []
    positive_rec = []
    negative_rec = []

    ######### to test each epoch ###############################################################
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

    ############################################################################################

    # another test for git
    for epoch in range(resume_epoch, opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        for i in range(opt.model_number):
            m = model_ls[i]
            op = opt_ls[i]
            train(train_loader=tl_ls[i], encoder=m[0], decoder=m[1], criterion=criterion, encoder_optim=op[0],
                  decoder_optim=op[1], epoch=epoch + 1, opt=opt, num=i + 1)
            s = scheduler_ls[i]
            s[0].step()
            s[1].step()

        ##################################
        # save checkpoints
        ##################################
        if ((epoch + 1) % opt.epoch_save) == 0:
            for i in range(opt.model_number):
                m = model_ls[i]
                encoder_path_checkpoint = '{0}/{1}_state_epoch{2}_encoder_model_{3}.pth'.format(opt.checkpoint_folder, opt.prefix,
                                                                               epoch + 1, i + 1)
                utils.save_checkpoint(m[0].state_dict(), encoder_path_checkpoint)

                decoder_path_checkpoint = '{0}/{1}_state_epoch{2}_decoder_model_{3}.pth'.format(opt.checkpoint_folder, opt.prefix,
                                                                               epoch + 1, i + 1)
                utils.save_checkpoint(m[1].state_dict(), decoder_path_checkpoint)

                print('Save encoder model to {0}'.format(encoder_path_checkpoint))
                print('Save decoder model to {0}'.format(decoder_path_checkpoint))


        if ((epoch + 1) % opt.epoch_plot) == 0:
            plt.figure(1)
            # plt.subplot(1, 2, 1)
            plt.plot(loss_rec)
            plt.legend('loss')
            # plt.subplot(1, 2, 2)
            # plt.plot(positive_rec)
            # plt.plot(negative_rec)
            # plt.legend(('simmilarity of positives', 'simmilarity of negatives'))
            plt.show()
            plt.savefig('./figures/lstm_result{0}.jpg'.format(epoch + 1))
            plt.close()
        if ((epoch + 1) % opt.epoch_test) == 0:
            evaluate.test(test_video_loader, test_audio_loader, model_ls, opts_test)


if __name__ == '__main__':
    main()
