import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import torchvision
import torch


class VideoFeatDataset(data.Dataset):
    def __init__(self, root, flist=None, frame_num=120, which_feat='both', CompressRatio=2):
        self.root = root
        self.pathlist = self.get_file_list(flist)
        self.fnum = frame_num
        self.which_feat = which_feat
        self.CompressRatio = CompressRatio

    def __getitem__(self, index):
        path = os.path.join(self.root, self.pathlist[index])
        if self.which_feat == 'vfeat':
            path = os.path.join(path, 'frames')
            ls = os.listdir(path)
            first = 1
            for i in ls:
                img_path = os.path.join(path, i)
                img = Image.open(img_path)
                img = img.resize((260,360))
                #img = img.resize(((img.size[0] / self.CompressRatio), (img.size[1] / self.CompressRatio)))
                totensor = torchvision.transforms.ToTensor()
                img = totensor(img)
                #img = np.array(img).transpose()
                img = img.resize_(1, img.shape[0], img.shape[1], img.shape[2])
                if first:
                    first = 0
                    vfeat = img
                else:
                    vfeat = torch.cat((vfeat,img),dim=0)

            return vfeat

        elif self.which_feat == 'afeat':
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')  # audio feature
            afeat = torch.from_numpy(afeat)

            return afeat

        else:
            # load each frames of the video
            path_v = os.path.join(path, 'frames')
            ls = os.listdir(path_v)
            first = 1
            for i in ls:
                img_path = os.path.join(path_v, i)
                img = Image.open(img_path)
                img = img.resize((260,360))
                # img = img.resize(((img.size[0] / self.CompressRatio), (img.size[1] / self.CompressRatio)))
                totensor = torchvision.transforms.ToTensor()
                img = totensor(img)
                #img = img.resize(((img.size[0] / self.CompressRatio), (img.size[1] / self.CompressRatio)))
                #img = np.array(img).transpose()
                img = img.resize_(1, img.shape[0], img.shape[1], img.shape[2])
                if first:
                    first = 0
                    vfeat = img
                else:
                    vfeat = torch.cat((vfeat,img),dim=0)
            # load the afeat.npy
            afeat = self.loader(os.path.join(path, 'afeat.npy')).astype('float32')  # audio feature
            afeat = torch.from_numpy(afeat)

            return vfeat,afeat
    def __len__(self):
        return len(self.pathlist)

    def loader(self, filepath):
        return np.load(filepath)

    def get_file_list(self, flist):
        filelist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                filepath = line.strip()
                filelist.append(filepath)
        return filelist


