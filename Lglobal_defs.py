# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: some global definitions. This script should
            include no other scripts in the project.
"""

import os
from enum import IntEnum


join = os.path.join
exists = os.path.exists


def mk_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)
    return name


# top directories
PATH_BDATASET = '../../Bdataset'

#ml_prj/Lsaving/Ldata_helper.py
try:
    from modelarts.session import Session
    PATH_SAVING = mk_dir('work/Lsaving')
except:
    PATH_SAVING = mk_dir('Lsaving')

PATH_CV = join(PATH_SAVING, 'cv_files')
PATH_MNIST = join(PATH_SAVING, 'MNIST_data')
PATH_fer2013 = join(PATH_BDATASET,'fer2013','fer2013.csv')
PATH_FORMATTED_FER2013 = mk_dir(join(PATH_SAVING, 'format_fer2013'))
PATH_FIGS = mk_dir(join(PATH_SAVING, 'figs'))


#exists(PATH_FORMATTED_FER2013)= True
#
#s3://obs-optimusling2/ml_prj/
if __name__=='__main__':
    print("exists(PATH_SAVING)=",exists(PATH_SAVING))
    print("exists(PATH_FORMATTED_FER2013)=",exists(PATH_FORMATTED_FER2013))
    #print(exists(PATH_BDATASET))
    #print(exists(PATH_fer2013))

"""
#print("exists(url)=",exists("https://obs-optimusling2.obs.cn-north-1.myhuaweicloud.com/ml_prj"))
print("exists(PATH_SAVING)=",exists(PATH_SAVING))
print("exists(PATH_FORMATTED_FER2013)=",exists(PATH_FORMATTED_FER2013))
print("exists(Lsaving/format_fer2013)=",exists("Lsaving/format_fer2013"))
#print("os.listdir(Lsaving/format_fer2013)",os.listdir('Lsaving/format_fer2013'))
print("exists(Lsaving/format_fer2013/fer2013)=",exists("Lsaving/format_fer2013/fer2013"))
print("exists(ml_prj/Lsaving/format_fer2013/fer2013.h5)=",exists("ml_prj/Lsaving/format_fer2013/fer2013.h5"))
print("exists(Lsaving/format_fer2013/Ldata_helper.py)=",exists("Lsaving/format_fer2013/Ldata_helper.py"))
print("exists(Lsaving/Ldata_helper.py)=",exists("Lsaving/Ldata_helper.py"))
print("exists(Ldata_helper.py)=",exists("Ldata_helper.py"))
#s3://obs-optimusling2/ml_prj/
print("exists(s3://obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)=",exists("s3://obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py"))
print("exists(s3://obs-optimusling2/ml_prj/Lsaving)=",exists("s3://obs-optimusling2/ml_prj/Lsaving"))
print("exists(s3://obs-optimusling2)=",exists("s3://obs-optimusling2"))
print("exists(s3://obs-optimusling2/fer2013.h5)=",exists("s3://obs-optimusling2/fer2013.h5"))
print("exists(s3://obs-optimusling2/images)=",exists("s3://obs-optimusling2/images"))
print("exists(s3://obs-optimusling2/data)=",exists("s3://obs-optimusling2/data"))
print("exists(s3://obs-optimusling2/data/fer2013.h5)=",exists("s3://obs-optimusling2/data/fer2013.h5"))
print("exists(s3://obs-optimusling2/data/images)=",exists("s3://obs-optimusling2/data/images"))
print("exists(s3://obs-optimusling2/ml_prj)=",exists("s3://obs-optimusling2/ml_prj"))

print("exists(/obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)=",exists("/obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py"))
print("exists(/obs-optimusling2/ml_prj/Lsaving)=",exists("/obs-optimusling2/ml_prj/Lsaving"))
print("exists(/obs-optimusling2/ml_prj)=",exists("/obs-optimusling2/ml_prj"))
print("exists(/obs-optimusling2)=",exists("/obs-optimusling2"))
print("exists(/obs-optimusling2/fer2013.h5)=",exists("/obs-optimusling2/fer2013.h5"))
print("exists(/obs-optimusling2/images)=",exists("/obs-optimusling2/images"))
print("exists(/obs-optimusling2/data)=",exists("/obs-optimusling2/data"))
print("exists(/obs-optimusling2/data/fer2013.h5)=",exists("/obs-optimusling2/data/fer2013.h5"))
print("exists(/obs-optimusling2/data/images)=",exists("/obs-optimusling2/data/images"))


print("exists(obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)=",exists("obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py"))
print("exists(obs-optimusling2/ml_prj/Lsaving)=",exists("obs-optimusling2/ml_prj/Lsaving"))
print("exists(obs-optimusling2/ml_prj)=",exists("obs-optimusling2/ml_prj"))
print("exists(obs-optimusling2)=",exists("obs-optimusling2"))
print("exists(obs-optimusling2/fer2013.h5)=",exists("obs-optimusling2/fer2013.h5"))
print("exists(obs-optimusling2/images)=",exists("obs-optimusling2/images"))
print("exists(obs-optimusling2/data)=",exists("obs-optimusling2/data"))
print("exists(obs-optimusling2/data/fer2013.h5)=",exists("obs-optimusling2/data/fer2013.h5"))
print("exists(obs-optimusling2/data/images)=",exists("obs-optimusling2/data/images"))


print("exists(ml_prj/Lsaving/Ldata_helper.py)=",exists("ml_prj/Lsaving/Ldata_helper.py"))
print("exists(ml_prj/Lsaving)=",exists("ml_prj/Lsaving"))
print("exists(ml_prj)=",exists("ml_prj"))
print("exists(obs-optimusling2)=",exists("obs-optimusling2"))
print("exists(fer2013.h5)=",exists("fer2013.h5"))
print("exists(images)=",exists("images"))
print("exists(data)=",exists("data"))
print("exists(data/fer2013.h5)=",exists("data/fer2013.h5"))
print("exists(data/images)=",exists("data/images"))


print("exists(obs-optimusling2)=",exists("obs-optimusling2"))

print("exists(ml_prj)=",exists("ml_prj"))
print("exists(obs-optimusling2/ml_prj)=",exists("obs-optimusling2/ml_prj"))
print("exists(ml)=",exists("ml"))
#exists(PATH_SAVING)= True


exists(PATH_SAVING)= True
exists(PATH_FORMATTED_FER2013)= True
exists(Lsaving/format_fer2013)= True
os.listdir(Lsaving/format_fer2013) []
exists(Lsaving/format_fer2013/fer2013)= False
exists(Lsaving/format_fer2013/fer2013.h5)= False
exists(Lsaving/format_fer2013/Ldata_helper.py)= False
exists(Lsaving/Ldata_helper.py)= False
exists(Ldata_helper.py)= False
exists(s3://obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)= False
exists(s3://obs-optimusling2/ml_prj/Lsaving)= False
exists(s3://obs-optimusling2)= False
exists(s3://obs-optimusling2/fer2013.h5)= False
exists(s3://obs-optimusling2/images)= False
exists(s3://obs-optimusling2/data)= False
exists(s3://obs-optimusling2/data/fer2013.h5)= False
exists(s3://obs-optimusling2/data/images)= False
exists(s3://obs-optimusling2/ml_prj)= False
exists(/obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)= False
exists(/obs-optimusling2/ml_prj/Lsaving)= False
exists(/obs-optimusling2/ml_prj)= False
exists(/obs-optimusling2)= False
exists(/obs-optimusling2/fer2013.h5)= False
exists(/obs-optimusling2/data)= False
exists(/obs-optimusling2/data/fer2013.h5)= False
exists(/obs-optimusling2/data/images)= False
exists(obs-optimusling2/ml_prj/Lsaving/Ldata_helper.py)= False
exists(obs-optimusling2/ml_prj/Lsaving)= False
exists(obs-optimusling2/fer2013.h5)= False
exists(obs-optimusling2/images)= False
exists(obs-optimusling2/data)= False
exists(obs-optimusling2/data/fer2013.h5)= False
exists(obs-optimusling2/data/images)= False
exists(ml_prj/Lsaving/Ldata_helper.py)= True
exists(ml_prj/Lsaving)= True
exists(ml_prj)= True
exists(obs-optimusling2)= False
exists(fer2013.h5)= False
exists(images)= False
exists(data)= False
exists(data/fer2013.h5)= False
exists(data/images)= False
exists(obs-optimusling2)= False
exists(ml_prj)= True
exists(obs-optimusling2/ml_prj)= False
exists(ml)= False
exists(PATH_SAVING)= True
exists(PATH_FORMATTED_FER2013)= True

"""