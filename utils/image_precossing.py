import PIL
from PIL import Image, ImageOps
import numpy as np
import torch

def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
    return tensor.view((1, ) + tensor.size())


def _remove_batch(tensor):
    """
    Return a tensor with size tensor.size()[1:]
    :param tensor: 3D or 4D tensor
    :return: 2D or 3D tensor
    """
    return tensor.view(tensor.size()[1:])


def resize_images(images, resize=128):
    B, C, W, H = images.shape
    results = []
    for i in range(B):
        I = (np.transpose(images[i], axes=[1, 2, 0]) * 255).astype(np.uint8)
        I = Image.fromarray(I)
        I = I.resize((resize, resize))
        I = np.array(I).astype(np.float32) / 255.0
        results.append(np.expand_dims(np.transpose(I, axes=[2, 0, 1]), axis=0))
    results = np.concatenate(results, axis=0)
    return results

def post_process_image(x):
    return torch.clamp(_tanh_to_sigmoid(x), min=0.0, max=1.0)
