import argparse
import os
from functools import partial

import numpy as np
import torch
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.auc_utils.utils import load_images

LPIPS = None


@torch.no_grad()
def calculate_lpips(imgs1, imgs2):
    """ Nx3xHxW, [-1, 1] """
    imgs1 = torch.tensor(np.array(imgs1)).permute(0, 3, 1, 2)
    imgs2 = torch.tensor(np.array(imgs2)).permute(0, 3, 1, 2)
    assert imgs1.shape[1] == 3
    assert imgs1.shape == imgs2.shape
    global LPIPS
    if LPIPS is None:
        LPIPS = lpips.LPIPS()
    res = LPIPS.forward(imgs1, imgs2).squeeze()
    res = res.cpu().numpy()
    return res.mean()


# https://arxiv.org/pdf/1603.08511.pdf
# Raw accuracy (AuC): As a low-level test, we compute the percentage
# of predicted pixel colors within a thresholded L2 distance of the ground truth
# in ab color space. We then sweep across thresholds from 0 to 150 to produce
# a cumulative mass function, as introduced in [22], integrate the area under the
# curve (AuC), and normalize. Note that this AuC metric measures raw prediction
# accuracy, whereas our method aims for plausibility
def calculate_auc(i1, i2):
    i1 = color.rgb2lab(i1)
    i2 = color.rgb2lab(i2)
    i1 = i1[:, :, 1:]
    i2 = i2[:, :, 1:]
    distance_matrix = np.linalg.norm(i1 - i2, axis=2)
    count = 0
    n = 150
    for i in range(0, n):
        count += np.sum(distance_matrix <= i)
    auc = count / (n * distance_matrix.shape[0] * distance_matrix.shape[1])
    return auc


def calculate_metric(imgs1, imgs2, metric):
    if metric == 'lpips':
        return calculate_lpips(imgs1, imgs2)
    if metric == 'psnr':
        metric_evaluator = peak_signal_noise_ratio
    elif metric == 'ssim':
        metric_evaluator = partial(structural_similarity, multichannel=True)
    elif metric == 'auc':
        metric_evaluator = calculate_auc
    else:
        raise NotImplementedError
    assert len(imgs1) == len(imgs2)
    results = []
    for i1, i2 in zip(imgs1, imgs2):
        res = metric_evaluator(i1, i2)
        results.append(res)
    results = np.array(results)
    return results.mean()


def main(args):
    metrics = [args.metric]
    if args.metric == 'all':
        metrics = ['lpips', 'psnr', 'ssim', 'auc']
    models = os.listdir(args.result_path)
    for model in models:
        datasets = os.listdir(os.path.join(args.result_path, model))
        for dataset in datasets:
            src_img_dir = os.path.join(args.origin_path, dataset)
            trg_img_dir = os.path.join(args.result_path, model, dataset)
            src_imgs = load_images(src_img_dir)
            trg_imgs = load_images(trg_img_dir)
            for metric in metrics:
                res = calculate_metric(src_imgs, trg_imgs, metric)
                print(f'{metric.upper()} for {model.upper()} on {dataset.upper()} is {res:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_path', type=str, default='./data/origin')
    parser.add_argument('--result_path', type=str, default='./data/inversion')
    parser.add_argument('--metric', type=str, choices=['all', 'lpips', 'psnr', 'ssim', 'auc'], default='all')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    main(parser.parse_args())
