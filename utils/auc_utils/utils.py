import glob
import os

import numpy as np
from PIL import Image


def name2int(p):
    return int(os.path.basename(p).split('.')[0])


def load_images(img_dir):
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_paths.extend(glob.glob(os.path.join(img_dir, '*.png')))
    img_paths.sort(key=name2int)
    img_list = []
    for img_path in img_paths:
        img_list.append(np.array(Image.open(img_path)))
    return img_list
