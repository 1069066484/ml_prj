# -*- coding: utf-8 -*-
from modelarts.session import Session
#session = Session()
#path_local = '/code_caffe'
#session.download_data(bucket_path="obs-optimusling2/code_caffe/README.md", path=path_local)
#session.upload_data(bucket_path="obs-optimusling2/code_caffe", path=path)
import shutil
import zipfile
import os


def unzip():
    file_list = os.listdir(r'.')

    for file_name in file_list:
        if os.path.splitext(file_name)[1] == '.zip':
            print(file_name)
            file_zip = zipfile.ZipFile(file_name, 'r')
            for file in file_zip.namelist():
                file_zip.extract(file, r'.')
            file_zip.close()
            os.remove(file_name)    

            
def zip(file_list, dst_folder_name):
    '''
    批量复制文件到指定文件夹，然后把指定文件夹的内容压缩成ZIP并且删掉该文件夹
    :param file_list: 文件或文件夹
    :param dst_folder_name: 目标压缩文件的名称
    :return:
    '''
    for fn in file_list:
        if not os.path.exists(fn):
            raise Exception( fn + ''' doesn't exist''')
    for item in file_list:
        copy_file(item, dst_folder_name)
    # 这里我把输出文件的路径选到了程序根目录下
    source = os.getcwd() + "\\" + dst_folder_name
    shutil.make_archive(source, "zip", source)
    shutil.rmtree(source)
 
 
def copy_file(srcfile, filename):
    '''
    把文件或文件夹复制到指定目录中
    :param srcfile: 文件或者文件夹的绝对路径
    :param filename: 指定目录
    :return:
    '''
    dstfile = os.path.abspath(os.getcwd())
    # 这里我把输出文件的路径选到了程序根目录下
    folder_name = dstfile + "\\" + filename + "\\"
    if not os.path.isfile(srcfile):
        last_name = os.path.basename(srcfile)
        destination_name = folder_name + last_name
        shutil.copytree(srcfile, destination_name)
        print("copy %s -> %s" % (srcfile, destination_name))
    else:
        fpath, fname = os.path.split(folder_name)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copy2(srcfile, folder_name)  # 移动文件
        print("copy %s -> %s" % (srcfile, folder_name))
        

def zip2(startdir,file_news):
    file_news = startdir +'.zip' # 压缩后文件夹的名字
    z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) #参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(startdir):
        fpath = dirpath.replace(startdir,'') #这一句很重要，不replace的话，就从根目录开始复制
        fpath = fpath and fpath + os.sep or ''#这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
            # print ('压缩成功')
    z.close()

            
if __name__=='__main__':
    unzip()