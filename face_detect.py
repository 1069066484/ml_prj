import sys
from Lglobal_defs import *
import cv2
from Ldata_helper import *
import numpy as np


path_cascade = join(PATH_CV,  'haarcascade_frontalface_default.xml')
cascade = cv2.CascadeClassifier(path_cascade)


def cv_enhance(data, only_one):
    # print(len(data))
    data = data[0]
    # print(type(data))
    enhanced = [data.copy()]
    data = data.reshape([48,-1]).T
    data = np.array((data,data,data), np.uint8)
    data = np.swapaxes(data,0,2).copy()
    faces = cascade.detectMultiScale(data, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in faces:
        res = data[x:x+w,y:y+h,0]
        res = cv2.resize(res, (48,48))
        enhanced.append(res.reshape(-1))
    if only_one:
        enhanced = [enhanced[-1]]
    return enhanced


def cv_show(data):
    data = data.reshape([48,-1]).T
    data = np.array((data,data,data), np.uint8)
    data = np.swapaxes(data,0,2).copy()
    print("SHOW")
    cv2.imshow("1", data)
    cv2.waitKey()


def read_d1():
    path = join(PATH_CV,  npfn('test'))
    if exists(path):
        return np.load(path)
    [tr_dl, te_dl] = read_format_fer2013_dls()
    np.save(path, te_dl[0][5])
    return te_dl[0][5]


def test():
    path_img = join(PATH_SAVING, 'images/1.jpg')
    path_cascade = join(PATH_CV,  'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(path_cascade)
    img = cv2.imread(path_img, cv2.COLOR_BGR2GRAY)

    data = read_d1().reshape([48,48]).T
    data = np.array((data,data,data), np.uint8)
    data = np.swapaxes(data,0,2).copy()
    #print(data[0:3,0:3,0:3])
    # print(img[0:2,0:2,0:2])
    print(data.shape, type(data))
    # data = cv2.resize(data, (100,100))

    #print(data[0:3,0:3,0:3])
    #(data.shape, type(data))
    #(48, 48, 3)
    #(40, 40, 3)

    img = data

    faces = cascade.detectMultiScale(img.copy(), scaleFactor=1.1, minNeighbors=3)

    print(len(faces),'faces are found')
    for x, y, w, h in faces:
        #cv2.circle(img,((x+x+w)//2,(y+y+h)//2),w//2,(0,255,0),2)
        cv2.rectangle(img, (x,y), (x+w,y+h),1)
    res = img[x:x+w,y:y+h,0]
    res = cv2.resize(res, (48,48))
    print(res.shape)
    cv2.imshow('1', img)
    cv2.waitKey()


if __name__=='__main__':
    #test()
    d = read_d1()
    #print(d[:80])
    en = cv_enhance(d)
    for e in en:
        cv_show(e)