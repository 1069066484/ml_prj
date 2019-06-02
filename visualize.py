import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as functional
from Lglobal_defs import *
from Ldata_helper import *
from torch.autograd import Variable
from torchvision import transforms
from skimage import io
from skimage.transform import resize
import VGG_torch as VGG


cut_size = 44

trans_te = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops]))
    ]) 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [.299,.587,.114])

#raw_img = io.read(join(PATH_SAVING, 'images/1.jpg'))
raw_img = io.imread(join(PATH_SAVING, 'images/1.jpg'))
gray = rgb2gray(raw_img)
gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
img = gray[:,:,np.newaxis]

img = np.concatenate((img , img , img ), axis=2)
img = Image.fromarray(img)
inputs = trans_te(img)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG.VGG(VGG.VGGType.VGG19)

path = mk_dir(join(PATH_SAVING, 'training_log'))
checkpoint = torch.load(join(path, 'priT_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

ncrops, c, h, w = np.shape(inputs)

inputs = inputs.view(-1, c, h, w)
inputs = inputs.cuda()
with torch.no_grad():
    inputs = Variable(inputs)
outputs = net(inputs)
outputs_avg = outputs.view(ncrops, -1).mean(0)
score = functional.softmax(outputs_avg).data.cpu().numpy()

_, pred = torch.max(outputs_avg.data, 0)
plt.rcParams['figure.figsize'] = (13.5,13.5)

axes = plt.subplot(1,2,1)
plt.imshow(raw_img)
plt.xlabel('Input Image -- classified as ' + str(class_names[int(pred.cpu().numpy())]))
plt.tight_layout()

plt.subplot(1,2,2)
ind = .1+.6*np.arange(len(class_names))
width = .4


for i in range(len(class_names)):
    plt.bar(ind[i], score[i], width)

plt.title('Classification Results')
plt.xlabel('Expression Category')
plt.ylabel('Classification Score')
plt.xticks(ind, class_names, rotation=45, fontsize=14)




plt.show()


print("Prediction is", str(class_names[int(pred.cpu().numpy())]))












