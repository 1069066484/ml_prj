from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import Ldata_helper as data_helper


class Fer2013Dataset:
    def __init__(self, split='Training'):
        self.split = split.lower()
        h5data = data_helper.read_format_fer2013_h5()
        if self.split == 'training':
            self.data = np.asarray(h5data['tr_d']).reshape([-1, 48,48])
            self.labels = h5data['tr_l']
        elif self.split == 'pubt':
            self.data = np.asarray(h5data['pubT_d']).reshape([-1, 48,48])
            self.labels = h5data['pubT_l']
        elif self.split == 'prit':
            self.data = np.asarray(h5data['priT_d']).reshape([-1, 48,48])
            self.labels = h5data['priT_l']


if __name__=='__main__':
    ds = Fer2013Dataset()
    img, target = ds[2]
    print(img.shape, target.shape)
    img, target = ds[2:10]
    print(img.shape, target.shape)
    """
    (48, 48, 3) ()
    (8, 48, 3, 48) (8,)
    """