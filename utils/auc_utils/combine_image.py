import argparse
import glob
import os

import numpy as np
from PIL import Image

from utils import name2int


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    models = os.listdir(args.input_path)
    for model in models:
        datasets = os.listdir(os.path.join(args.input_path, model))
        for dataset in datasets:
            all_imgs = glob.glob(os.path.join(args.input_path, model, dataset, '*.jpg'))
            all_imgs.extend(glob.glob(os.path.join(args.input_path, model, dataset, '*.png')))
            all_imgs.sort(key=name2int)
            for i in range(0, len(all_imgs), args.num):
                j = i + args.num
                img_paths = all_imgs[i:j]
                imgs = []
                for img_path in img_paths:
                    imgs.append(np.array(Image.open(img_path)))
                combined_img = np.concatenate(imgs, axis=1)
                img = Image.fromarray(combined_img)
                img.save(os.path.join(args.output_path, f"{model}_{dataset}_{i}~{j}.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/interpolation')
    parser.add_argument('--output_path', type=str, default='./data/combined_interpolation')
    parser.add_argument('--num', type=int, default=10)
    main(parser.parse_args())
