# -*- coding: utf-8 -*-
import csv
import os
from PIL import Image
import numpy as np
from Lglobal_defs import *
from Ldata_helper import *
import os


datasets_path = PATH_BDATASET


train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')



def read_format_fer2013_dls():
    f = csv.reader(open(global_defs.PATH_fer2013))
    train_file = open(train_csv,'w', newline='')
    train_csv_writer = csv.writer(train_file,dialect='excel')
    val_file = open(val_csv,'w', newline='')
    val_csv_writer = csv.writer(val_file,dialect='excel')
    test_file = open(test_csv,'w', newline='')
    test_csv_writer = csv.writer(test_file,dialect='excel')
    for label, data1, trte1  in f:
        #data1 = data1.split(' ')
        if trte1.startswith('Tr'):
            train_csv_writer.writerow([label, data1])
        elif trte1.startswith('Pub'):
            val_csv_writer.writerow([label, data1])
        elif trte1.startswith('Pri'):
            test_csv_writer.writerow([label, data1])
    train_file.close()
    val_file.close()
    test_file.close()


#read_format_fer2013_dls()
#exit(0)

train_set = os.path.join(datasets_path, 'train')
val_set = os.path.join(datasets_path, 'val')
test_set = os.path.join(datasets_path, 'test')

for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = 1
    with open(csv_file) as f:
        csvr = csv.reader(f)
        #header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            subfolder = os.path.join(save_path, label)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            im = Image.fromarray(pixel).convert('L')
            image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
            print(image_name)
            im.save(image_name)
