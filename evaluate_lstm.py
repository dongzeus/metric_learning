from __future__ import print_function
from __future__ import division
import os
from optparse import OptionParser
from tools.config_tools import Config

# ----------------------------------- loading paramters -------------------------------------------#
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)
# --------------------------------------------------------------------------------------------------#

# ------------------ environment variable should be set before import torch  -----------------------#
if opt.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('setting gpu on gpuid {0}'.format(opt.gpu_id))
# --------------------------------------------------------------------------------------------------#

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import models_lstm as models
from dataset_lstm import VideoFeatDataset as dset
from tools import utils

# reminding the cuda option
if torch.cuda.is_available():
    if not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    else:
        cudnn.benchmark = True

# loading test dataset
test_video_dataset = dset(root=opt.data_dir, flist=opt.video_flist, which_feat='vfeat', creat_test=0)
test_audio_dataset = dset(root=opt.data_dir, flist=opt.audio_flist, which_feat='afeat', creat_test=0)
print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')

def pca_tensor(tensor, pr=False):
    bs = tensor.size(0)
    pca = PCA(n_components=opt.afeat_pca)
    if isinstance(tensor, Variable):
        tensor_np = tensor.data.numpy()
    else:
        tensor_np = tensor.numpy()
    tensor_np.resize(bs * 120, 128)
    pca.fit(tensor_np)
    tensor_np = pca.transform(tensor_np)
    tensor_np.resize(bs, 120, opt.afeat_pca)
    tensor_new = torch.from_numpy(tensor_np)
    if isinstance(tensor,Variable):
        tensor_new = Variable(tensor_new)
        if opt.cuda:
            tensor_new = tensor_new.cuda()

    if pr:
        print('afeat PCA variance ratio:')
        print(pca.explained_variance_ratio_)
        print(np.sum(pca.explained_variance_ratio_))
    return tensor_new


# test function for metric learning
def test(video_loader, audio_loader, model_ls, opt):
    for num, m in enumerate(model_ls):

        encoder = m[0]
        decoder = m[1]

        # evaluation mode: only useful for the models with batchnorm or dropout
        encoder.eval()
        decoder.eval()

        right = 0  # correct sample number

        # audio_gen is the audio feature generate by encoder-decoder model from video feature
        for i, vfeat in enumerate(video_loader):
            bs = vfeat.size()[0]
            vfeat_var = Variable(vfeat, volatile=True)
            if opt.cuda:
                vfeat_var = vfeat_var.cuda()
            encoder_hidden = encoder.init_hidden(batch_size=bs)
            encoder_output, encoder_hidden = encoder(vfeat_var, encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_input = encoder_output[:, 119, :]  # bs * 128
            decoder_context = torch.mean(encoder_output, dim=1)  # bs * 128

            # Generate the audio from video feature
            for seq in range(120):
                audio_output, decoder_context, decoder_hidden, attn_weights = decoder(decoder_input, decoder_context,
                                                                                      decoder_hidden, encoder_output)
                decoder_input = audio_output
                if seq == 0:
                    audio_gen_i = audio_output.view(bs, 1, -1).clone()
                else:
                    audio_gen_i = torch.cat((audio_gen_i, audio_output.view(bs, 1, -1)), dim=1)
            if i == 0:
                audio_gen = audio_gen_i.clone()
            else:
                audio_gen = torch.cat((audio_gen, audio_gen_i), dim=1)

        # audio_target is the ground truth of audio
        for j, afeat in enumerate(audio_loader):
            afeat = pca_tensor(afeat, pr=True)
            afeat_var = Variable(afeat, volatile=True)
            if opt.cuda:
                afeat_var = afeat_var.cuda()
            if j == 0:
                audio_target = afeat_var.clone()
            else:
                audio_target = torch.cat((audio_target, afeat_var), dim=0)

        # sim is the similarity matrix between vfeat and afeat
        num_test = audio_gen.size(0)
        for k in range(num_test):
            audio_gen_k = audio_gen[k, :, :].clone()
            audio_gen_k = audio_gen_k.repeat(num_test, 1, 1)

            sim_k = torch.nn.functional.pairwise_distance(audio_gen_k.view(num_test * audio_gen_k.size(1), -1),
                                                          audio_target.view(num_test * audio_target.size(1), -1))
            sim_k = sim_k.view(num_test, -1)
            sim_k = torch.mean(sim_k, dim=1)

            if k == 0:
                sim_k = sim_k.view(1, -1)
                sim = sim_k.clone()
            else:
                sim_k = sim_k.view(1, -1)
                sim = torch.cat((sim, sim_k), dim=0)

        # if your metric is the feature distance, you should set descending=False, else if your metric is feature similarity, you should set descending=True
        sorted, indices = torch.sort(sim, dim=1, descending=True)
        np_indices = indices.cpu().data.numpy()
        topk = np_indices[:, 0:opt.topk]
        for k in np.arange(num_test):
            order = topk[k, :]
            if k in order:
                right = right + 1
        print('==================================================================================')
        print('The No.{} similarity matrix: \n {}'.format(num + 1, sim))
        print('No.{} testing accuracy (top{}): {:.3f}'.format(num + 1, opt.topk, right / num_test))
        print('==================================================================================')

        if num == 0:
            simmat_ensem = sim
        else:
            simmat_ensem = simmat_ensem + sim

    sorted, indices = torch.sort(simmat_ensem, dim=1, descending=True)
    np_indices = indices.cpu().data.numpy()
    topk = np_indices[:, 0:opt.topk]
    right = 0
    num_test = simmat_ensem.size(0)
    for k in np.arange(num_test):
        order = topk[k, :]
        if k in order:
            right = right + 1
    print('==================================================================================')
    print('The ensembel similarity matrix: \n {}'.format(simmat_ensem))
    print('Ensembel testing accuracy (top{}): {:.3f}'.format(opt.topk, right / num_test))
    print('==================================================================================')


def main():
    global opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))
    # create model
    encoder = models.Encoder(batch_size=opt.batchSize)
    decoder = models.AttnDecoder()

    if opt.init_encoder != '':
        print('loading pretrained encoder model from {0}'.format(opt.init_encoder))
        encoder.load_state_dict(torch.load(opt.init_encoder))
    else:
        raise IOError('Please add your pretrained model path to init_encoder in config file!')

    if opt.init_decoder != '':
        print('loading pretrained decoder model from {0}'.format(opt.init_decoder))
        decoder.load_state_dict(torch.load(opt.init_decoder))
    else:
        raise IOError('Please add your pretrained model path to init_decoder in config file!')

    if opt.cuda:
        print('shift model to GPU .. ')
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    m = [encoder, decoder]
    m_ls = [m]

    test(test_video_loader, test_audio_loader, m_ls, opt)


if __name__ == '__main__':
    main()
