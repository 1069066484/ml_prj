from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import Ldata_helper as data_helper


class Fer2013Dataset(data.Dataset):
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split.lower()
        h5data = data_helper.read_format_fer2013_h5()
        if self.split == 'training':
            self.tr_d = np.asarray(h5data['tr_d']).reshape([-1, 48,48])
            self.tr_l = h5data['tr_l']
        elif self.split == 'pubt':
            self.pubT_d = np.asarray(h5data['pubT_d']).reshape([-1, 48,48])
            self.pubT_l = h5data['pubT_l']
        elif self.split == 'prit':
            self.priT_d = np.asarray(h5data['priT_d']).reshape([-1, 48,48])
            self.priT_l = h5data['priT_l']

    def __getitem__(self, index):
        if self.split == 'training':
            img, target = self.tr_d[index], self.tr_l[index]
        elif self.split == 'pubt':
            img, target = self.pubT_d[index], self.pubT_l[index]
        else:
            img, target = self.priT_d[index], self.priT_l[index]
        img = img[:,:,np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'training':
            return len(self.tr_d)
        elif self.split == 'pubt':
            return len(self.pubT_d)
        else:
            return len(self.priT_d)


