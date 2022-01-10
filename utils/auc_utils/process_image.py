import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_images


def processor(img_arrs, process, args):
    imgs = []
    for img_arr in img_arrs:
        img = Image.fromarray(img_arr)
        if process == 'downsample':
            origin_size = img.size[0]
            downsample_size = origin_size // args.downsample_scale
            img = img.resize((downsample_size, downsample_size)).resize((origin_size, origin_size))
        elif process == 'gray':
            img = img.convert('LA').convert('RGB')
        else:
            raise NotImplementedError
        imgs.append(np.array(img))
    return imgs


def main(args):
    processes = [args.process]
    if args.process == 'all':
        processes = ['downsample', 'gray']
    datasets = os.listdir(args.input_path)
    for process in processes:
        suffix = f"-x{args.downsample_scale}" if process == 'downsample' else ''
        output_path = args.input_path + f'-{process}{suffix}'
        for dataset in datasets:
            src_img_dir = os.path.join(args.input_path, dataset)
            src_imgs = load_images(src_img_dir)
            output_dir = os.path.join(output_path, dataset)
            os.makedirs(output_dir, exist_ok=True)
            res = processor(src_imgs, process, args)
            for i, arr in tqdm(enumerate(res)):
                img = Image.fromarray(arr)
                img.save(os.path.join(output_dir, f"{i}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/origin')
    parser.add_argument('--process', type=str, choices=['all', 'downsample', 'gray'], default='all')
    parser.add_argument('--downsample_scale', type=int, default=8)
    main(parser.parse_args())
