# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some help functions.
"""

import Lglobal_defs as global_defs
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import csv
from sklearn.manifold import TSNE
import gzip
from scipy.io import loadmat
import h5py
import face_detect


def read_gened_da_dls():
    xyz = np.random.uniform(0.0,5.0,[500, 3])
    xyz_ = xyz.copy()
    xyz_[:,0] += 1.0
    xyz_[:,1] -= 1.0
    #xyz_trans = 


def read_usps_dl():
    m1 = loadmat(global_defs.PATH_USPS1)
    m2 = loadmat(global_defs.PATH_USPS2)
    data = np.vstack([m1['traindata'], m2['testdata']])
    labels = np.vstack([m1['traintarg'], m2['testtarg']])
    data /= 2
    data += 0.5
    labels[labels < 0.0] = 0.0
    return [data, labels]


def read_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(os.path.join(global_defs.PATH_MNIST),one_hot=one_hot)


def read_mnist_dl():
    mnist = read_mnist()
    data = np.vstack([mnist.train.images, mnist.test.images])
    labels = np.vstack([mnist.train.labels, mnist.test.labels])
    return [data, labels]


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def npfilename(filename):
    return posfix_filename(filename, '.npy')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def csvfilename(filename):
    return posfix_filename(filename, '.csv')


def h5filename(filename):
    return posfix_filename(filename, '.h5')


npfn = npfilename
pkfn = pkfilename
csvfn = csvfilename
h5fn = h5filename


def csvfile2nparr(csvfn):
    csvfn = csvfilename(csvfn)
    csvfn = csv.reader(open(csvfn,'r'))
    def read_line(line):
        return [float(i) for i in line]
    m = [read_line(line) for line in csvfn]
    return np.array(m)


def read_labeled_features(csvfn):
    arr = csvfile2nparr(csvfn)
    data, labels = np.hsplit(arr,[-1])
    labels = labels.reshape(labels.size)
    return [data, labels]



def plt_show_it_data(it_data, xlabel='iterations', ylabel=None, title=None, do_plt_last=True):
    y = it_data
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    if do_plt_last:
        plt.text(x[-1], y[-1], y[-1])
    plt.show()


def plt_show_scatter(xs, ys, xlabel=None, ylabel=None, title=None):
    colors = ['r', 'y', 'k', 'g', 'b', 'm']
    num2plt = min(len(colors), len(xs))
    for i in range(num2plt):
        plt.scatter(x=xs[i], y=ys[i], c=colors[i], marker='.')
    plt.xlabel('' if xlabel is None else xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    plt.show()


def non_repeated_random_nums(nums, num):
    num = math.ceil(num)
    nums = np.random.permutation(nums)
    return nums[:num]


def index_split(num, percent1):
    percent1 = math.ceil(num * percent1)
    nums = np.random.permutation(num)
    return [nums[:percent1], nums[percent1:]]


def labeled_data_split(labeled_data, percent_train=0.6):
    np.random.seed(0)
    train_idx, test_idx = index_split(labeled_data[0].shape[0], percent_train)
    train_ld = [labeled_data[0][train_idx], labeled_data[1][train_idx]]
    test_ld = [labeled_data[0][test_idx], labeled_data[1][test_idx]]
    return [train_ld, test_ld]


def rand_arr_selection(arr, num):
    nonrep_rand_nums = non_repeated_random_nums(arr.shape[0], num)
    return [arr[nonrep_rand_nums], nonrep_rand_nums]


def labels2one_hot(labels):
    labels = np.array(labels, dtype=np.int)
    if len(labels.shape) == 1:
        minl = np.min(labels)
        labels -= minl
        maxl = np.max(labels) + 1
        r = range(maxl)
        return np.array([[1 if i==j else 0 for i in r] for j in labels])
    return labels


def read_format_fer2013_dls():
    path_tr_data = global_defs.join(global_defs.PATH_FORMATTED_FER2013, npfilename('tr_data'))
    path_tr_label = global_defs.join(global_defs.PATH_FORMATTED_FER2013, npfilename('tr_label'))
    path_te_data = global_defs.join(global_defs.PATH_FORMATTED_FER2013, npfilename('te_data'))
    path_te_label = global_defs.join(global_defs.PATH_FORMATTED_FER2013, npfilename('te_label'))
    if global_defs.exists(path_tr_data):
        return [[np.load(path_tr_data), np.load(path_tr_label)], [np.load(path_te_data), np.load(path_te_label)]]
    f = csv.reader(open(global_defs.PATH_fer2013))
    tr_labels = []
    te_labels = []
    tr_data = []
    te_data = []
    for label, data1, trte1  in f:
        data1 = data1.split(' ')
        if trte1.startswith('Tr'):
            tr_labels.append(int(label))
            tr_data.append([int(i) for i in data1])
        elif trte1.startswith('Pub'):
            te_labels.append(int(label))
            te_data.append([int(i) for i in data1])
    tr_data = np.array(tr_data)
    te_data = np.array(te_data)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    np.save(path_tr_data, tr_data)
    np.save(path_te_data, te_data)
    np.save(path_tr_label, tr_labels)
    np.save(path_te_label, te_labels)
    return [[tr_data, tr_labels], [te_data, te_labels]]


def visualize_da(src_data, tgt_data_ori, tgt_data_adpted, title=None, figname=None):
    plt.figure(figsize=(15,10))
    if tgt_data_ori is None:
        visualize_da2(src_data, tgt_data_adpted, 'adapted', title, figname)
        return None
    elif tgt_data_adpted is None:
        visualize_da2(src_data, tgt_data_ori, 'original', title, figname)
        return None
    src_data, _ = rand_arr_selection(src_data, min(300, src_data.shape[0]))
    tgt_data_ori, _ = rand_arr_selection(tgt_data_ori, min(300, tgt_data_ori.shape[0]))
    tgt_data_adpted, _ = rand_arr_selection(tgt_data_adpted, min(300, tgt_data_adpted.shape[0]))
    div_idx1 = src_data.shape[0]
    div_idx2 = div_idx1 + tgt_data_ori.shape[0]
    tsne = TSNE(n_components=2, n_iter=500).fit_transform(np.vstack([src_data, tgt_data_ori, tgt_data_adpted]))
    plt.scatter(tsne[:div_idx1, 0], tsne[:div_idx1, 1], c='b', label='Source Data')
    plt.scatter(tsne[div_idx1:div_idx2, 0], tsne[div_idx1:div_idx2, 1], c='r', label='Target Data(Original)')
    plt.scatter(tsne[div_idx2:, 0], tsne[div_idx2:, 1], c='g', label='Target Data(Adapted)')
    plt.legend(loc = 'upper left')
    if title is not None:
        plt.title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)   


def visualize_img_bw(img_arr):
    side = int(round(img_arr.size ** 0.5))
    img_arr = img_arr.reshape(side,side)
    fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    sharex=True,
    sharey=True, )
    #ax = ax.flatten()
    ax.imshow(img_arr, cmap='Greys', interpolation='nearest')
    plt.tight_layout()
    plt.show()


def visualize_da2(src_data, tgt_data, tgt_label, title=None, figname=None):
    src_data, _ = rand_arr_selection(src_data, min(300, src_data.shape[0]))
    tgt_data, _ = rand_arr_selection(tgt_data, min(300, tgt_data.shape[0]))
    div_idx1 = src_data.shape[0]
    tsne = TSNE(n_components=2, n_iter=500).fit_transform(np.vstack([src_data, tgt_data]))
    plt.scatter(tsne[:div_idx1, 0], tsne[:div_idx1, 1], c='b', label='Source Data')
    plt.scatter(tsne[div_idx1:, 0], tsne[div_idx1:, 1], c='r', label='Target Data(' + tgt_label + ')')

    plt.legend(loc = 'upper left')
    if title is not None:
        plt.title(title)
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)   


def _test_visualize_da():
    a = np.array([[1,2,3],[5,4,1],[3,4,1],[6,9,1],[1,0,3]])
    labels = np.array([1,0,0,1,0])
    visualize_da(a[:2], a[2:] )


def shuffle_labeled_data(dl):
    data, labels = dl
    a = np.arange(labels.shape[0])
    np.random.seed(0)
    np.random.shuffle(a)
    return [data[a], labels[a]]


cv_detected = True
only_one = True
print('cv_detected:', cv_detected,'\tonly_one:',only_one)

def read_format_fer2013_h5():
    path_h5 = global_defs.join(global_defs.PATH_FORMATTED_FER2013, h5fn('fer2013' + ("_cv" if cv_detected else "") + 
                                                                        ("1" if (cv_detected and only_one) else "")))
    # print("exist=",path_h5,global_defs.exists(path_h5))
    if os.path.exists(path_h5):
        return h5py.File(path_h5, 'r', driver='core')
    csvf = csv.reader(open(global_defs.PATH_fer2013))
    tr_dl = [[],[]]
    pubT_dl = [[],[]]
    priT_dl = [[],[]]
    header = True
    for label, data, setname  in csvf:
        if header:
            header = False
            continue
        data = [int(i) for i in data.split(' ')]
        data = [np.array(data, dtype=np.uint8)]
        if (cv_detected and setname == 'Training') or (only_one and cv_detected):
            data = face_detect.cv_enhance(data, only_one)
        if setname == 'Training':
            tr_dl[0]+=data
            tr_dl[1]+=[int(label) for _ in range(len(data))]
        elif setname == 'PublicTest':
            pubT_dl[0]+=data
            pubT_dl[1]+=[int(label) for _ in range(len(data))]
        elif setname == 'PrivateTest':
            priT_dl[0]+=data
            priT_dl[1]+=[int(label) for _ in range(len(data))]
    print("collect: ", len(tr_dl[0]), len(pubT_dl[0]), len(priT_dl[1]))
    h5_datafile = h5py.File(path_h5, 'w')
    for dataset_name, dataset in [['tr', tr_dl], ['pubT', pubT_dl], ['priT', priT_dl]]:
        h5_datafile.create_dataset(dataset_name + '_d', dtype = 'uint8', data=dataset[0])
        h5_datafile.create_dataset(dataset_name + '_l', dtype = 'int64', data=dataset[1])
    h5_datafile.close()
    return h5py.File(path_h5, 'r', driver='core')



def _get_dicts_test():
    id2name, name2id = get_dicts()
    print(id2name, name2id)



def _test_labels_one_hot():
    a = np.array([2,1,0,0,0,2,1,1,1])
    print(labels2one_hot(a))


def _read_dataset_mini_test():
    p = r'G:\f\SJTUstudy\G3_SEMESTER2\machine_learning\prj\Bdataset\fer2013\t.csv'
    #print(os.path.exists(p))
    #exit(0)
    f = csv.reader(open(p))
    tr_labels = []
    te_labels = []
    tr_data = []
    te_data = []
    for label, data1, trte1  in f:
        data1 = data1.split(' ')
        if trte1.startswith('Tr'):
            #print(len(data1))
            tr_labels.append(int(label))
            tr_data.append([int(i) for i in data1])
        elif trte1.startswith('Te'):
            #print(len(data1))
            te_labels.append(int(label))
            te_data.append([int(i) for i in data1])
    tr_data = np.array(tr_data)
    print(tr_data[1])
    te_data = np.array(te_data)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    for a in [tr_data, te_data, tr_labels, te_labels]:
        print(a.shape)


def _read_format_fer2013_dls_test():
    train_dl, test_dl = read_format_fer2013_dls()
    print(train_dl[0].shape, train_dl[1].shape)
    print(test_dl[0].shape, test_dl[1].shape)


def _visualize_img_bw_test():
    train_dl, test_dl = read_format_fer2013_dls()
    i = 0
    while i < 50:
        i += 1
        visualize_img_bw(train_dl[0][i])


def _read_format_fer2013_h5_test():
    h5 = read_format_fer2013_h5()
    print(h5)


def _read_format_fer2013_h5_true_test():
    # read_format_fer2013_h5(True)
    read_format_fer2013_h5(True, True)


if __name__ == '__main__':
    _read_format_fer2013_h5_true_test()