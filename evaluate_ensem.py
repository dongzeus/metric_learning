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

import models_ensem as models
from dataset import VideoFeatDataset as dset
from tools import utils
from sklearn.decomposition import PCA

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


# test function for metric learning
def test(video_loader, audio_loader, model_ls, opt):
    for num, model in enumerate(model_ls):

        """
        train for one epoch on the training set
        """
        # evaluation mode: only useful for the models with batchnorm or dropout
        model.eval()

        right = 0  # correct sample number
        sample_num = 0  # total sample number
        sim_mat = []  # similarity matrix

        # ------------------------------------ important parameters -----------------------------------------------#
        # bz_sim: the batch similarity between two visual and auditory feature batches
        # slice_sim: the slice similarity between the visual feature batch and the whole auditory feature sets
        # sim_mat: the total simmilarity matrix between the visual and auditory feature datasets
        # -----------------------------------------------------------------------------------------------------#

        for i, vfeat in enumerate(video_loader):
            bz = vfeat.size()[0]
            sample_num += bz
            for j, afeat in enumerate(audio_loader):
                for k in np.arange(bz):

                    cur_vfeat = vfeat[k].clone()
                    cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                    vfeat_var = Variable(cur_vfeats, volatile=True)
                    afeat_var = Variable(afeat, volatile=True)
                    if opt.cuda:
                        vfeat_var = vfeat_var.cuda()
                        afeat_var = afeat_var.cuda()

                    cur_sim = model(vfeat_var, afeat_var)
                    cur_sim = cur_sim[:, 0].clone()
                    cur_sim = cur_sim.resize(bz, 1)
                    if k == 0:
                        bz_sim = cur_sim.clone()
                    else:
                        bz_sim = torch.cat((bz_sim, cur_sim.clone()), 1)
                if j == 0:
                    slice_sim = bz_sim.clone()
                else:
                    slice_sim = torch.cat((slice_sim, bz_sim.clone()), 0)
            if i == 0:
                sim_mat = slice_sim.clone()
            else:
                sim_mat = torch.cat((sim_mat, slice_sim.clone()), 1)

        # if your metric is the feature distance, you should set descending=False, else if your metric is feature similarity, you should set descending=True
        sorted, indices = torch.sort(sim_mat, 0, descending=True)
        np_indices = indices.cpu().data.numpy()
        topk = np_indices[:opt.topk, :]
        for k in np.arange(sample_num):
            order = topk[:, k]
            if k in order:
                right = right + 1
        print('==================================================================================')
        # print('The No.{} similarity matrix: \n {}'.format(num + 1, sim_mat))
        print('No.{} testing accuracy (top{}): {:.3f}'.format(num + 1, opt.topk, right / sample_num))
        print('==================================================================================')

        if num == 0:
            simmat_ensem = sim_mat
        else:
            simmat_ensem = simmat_ensem + sim_mat

    sorted, indices = torch.sort(simmat_ensem, 0, descending=True)
    np_indices = indices.cpu().data.numpy()
    topk = np_indices[:opt.topk, :]
    right = 0
    sample_num = simmat_ensem.size(0)
    for k in np.arange(sample_num):
        order = topk[:, k]
        if k in order:
            right = right + 1
    print('==================================================================================')
    # print('The ensembel similarity matrix: \n {}'.format(simmat_ensem))
    print('Ensembel testing accuracy (top{}): {:.3f}'.format(opt.topk, right / sample_num))
    print('==================================================================================')


def main():
    global opt
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt.batchSize,
                                                    shuffle=False, num_workers=int(opt.workers))
    # create model
    m_ls = []
    model = models.VA_lstm()

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))
    else:
        raise IOError('Please add your pretrained model path to init_model in config file!')

    if opt.cuda:
        print('shift model to GPU .. ')
        model = model.cuda()
    m_ls.append(model)
    test(test_video_loader, test_audio_loader, m_ls, opt)


if __name__ == '__main__':
    main()
