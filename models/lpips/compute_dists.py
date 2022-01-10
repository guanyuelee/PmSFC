import argparse
import models
from util import util
import os
import torch
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu,version='0.1')

# Load images
img0 = util.im2tensor(util.load_image(opt.path0)) # RGB image from [-1,1]
img1 = util.im2tensor(util.load_image(opt.path1))

if(opt.use_gpu):
    img0 = img0.cuda()
    img1 = img1.cuda()


# Compute distance
dist01 = model.forward(img0,img1)
print('Distance: %.3f'%dist01)

outputs = './experiment1'
original = os.path.join(outputs, 'target.png')
original_cuda = util.read_images_from_pathes([original], return_tensor=True).numpy()

files = ['baseline', 'classifier', 'discriminator', 'encoder']
maximum = 255

image_size = 64
h_num = 8
w_num = 4

for item in files:
    print('Read %s file. ' % item)
    cur_path = os.path.join(outputs, item)
    files = os.listdir(os.path.join(outputs, item))
    abs_files = [os.path.join(cur_path, f) for f in files]

    images = util.read_images_from_pathes(abs_files, return_tensor=True).numpy()

    lpips_list = []
    for i in range(4):
        whole_image = images[i:i+1]
        whole_target = original_cuda
        for h_i in range(h_num):
            for w_i in range(w_num):
                img0 = whole_image[:, :, h_i * image_size: (h_i + 1) * image_size,
                       w_i * image_size: (w_i + 1) * image_size]
                img1 = whole_target[:, :, h_i * image_size: (h_i + 1) * image_size,
                       w_i * image_size: (w_i + 1) * image_size]

                img0 = torch.from_numpy(img0 * 2 - 1.0)
                img1 = torch.from_numpy(img1 * 2 - 1.0)

                #img0 = img0.cuda()
                #img1 = img1.cuda()

                dist01 = model.forward(img1, img0).detach().numpy()
                lpips_list.append(dist01)

        print(files[i], '%.4f' % np.mean(lpips_list))
