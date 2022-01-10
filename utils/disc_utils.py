import numpy as np
import torch
import torch.nn as nn
from utils.file_utils import read_images_from_pathes
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, resize_images
import torchvision


class SimpleConv(nn.Module):
    def __init__(self, in_channel, out_channel, init=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1)
        if init:
            self.initialize()

    def initialize(self):
        error = 1e-5
        self.conv1.weight.data.fill_(error)
        self.conv1.bias.data.fill_(error)
        self.conv2.weight.data.fill_(error)
        self.conv2.bias.data.fill_(error)

    def forward(self, feature):
        return self.conv2(self.act(self.conv1(feature)))


def pixel_norm(x):
    x_sum = torch.sum(torch.pow(x, 2.0), dim=1, keepdim=True)
    x_sqrt = torch.sqrt(1e-8 + x_sum)
    return x / x_sqrt


def construct_forward_features(gen_block_ids, batch_size, dim_z, gen):
    z_train = torch.from_numpy(np.random.randn(batch_size, dim_z, 1, 1).astype(np.float32)).cuda()
    f_gens = []
    start = 0
    input_ = z_train
    for i in range(len(gen_block_ids)):
        end = gen_block_ids[i]
        f_gen = pixel_norm(gen([input_], which_block=None, free_model=True, start=start, end=end)).detach()
        f_gens.append(f_gen)
        start = end
        input_ = f_gen
    x_gen = gen([input_], which_block=gen_block_ids[-1], post_model=True).detach()
    return x_gen, f_gens


def construct_forward_features_given_z(z_noise, gen_block_ids, gen):
    f_gens = []
    start = 0
    input_ = z_noise
    for i in range(len(gen_block_ids)):
        end = gen_block_ids[i]
        f_gen = pixel_norm(gen([input_], which_block=None, free_model=True, start=start, end=end)).detach()
        f_gens.append(f_gen)
        start = end
        input_ = f_gen
    x_gen = gen([input_], which_block=gen_block_ids[-1], post_model=True).detach()
    return x_gen, f_gens


def construct_forward_features_given_f(f_feature, gen_block_ids, gen, if_detach=False):
    f_gens = []
    start = gen_block_ids[0]
    input_ = f_feature
    if if_detach:
        input_ = input_.detach()
    f_gens.append(input_)
    for i in range(len(gen_block_ids) - 1):
        end = gen_block_ids[i + 1]
        f_gen = pixel_norm(gen([input_], which_block=None, free_model=True, start=start, end=end)).detach()
        start = end
        input_ = f_gen
        if if_detach:
            input_ = input_.detach()
        f_gens.append(f_gen)

    x_gen = gen([input_], which_block=gen_block_ids[-1], post_model=True).detach()
    return x_gen, f_gens


def post_process_image(input_img, resize=256):
    test_img_rec = torch.clamp(_tanh_to_sigmoid(input_img), 0.0, 1.0).detach()
    test_img_rec = torch.from_numpy(resize_images(test_img_rec.cpu().numpy(), resize=resize))
    return test_img_rec


def compose_discriminator_forward(x_gen, disc, disc_block_ids, Cs, disc_detach, return_f_disc):
    f_disc, f_attns = disc([x_gen], which_block=disc_block_ids[0], pre_model=True, attain_list=disc_block_ids[1:])
    if disc_detach:
        f_disc_features = [f_disc.detach()] + [item.detach() for item in f_attns[::-1]]
    else:
        f_disc_features = [f_disc] + f_attns[::-1]

    f_gen_recs = []
    for i in range(len(f_disc_features)):
        if i == 0:
            f_gen_recs.append(pixel_norm(Cs[i](f_disc_features[i])))
        else:
            f_gen_recs.append(Cs[i](f_disc_features[i]))

    if return_f_disc:
        return f_gen_recs, f_disc_features

    return f_gen_recs


def report_reconstruction(images, gen, disc, Cs, gen_block_ids, disc_block_ids, file_name):
    test_img = read_images_from_pathes(images, return_cuda=True)
    test_img_reshape = torch.from_numpy(resize_images(test_img.detach().cpu().numpy(), resize=256))
    test_img_rec = batch_run(2, test_img * 2 - 1.0, easy_forward, gen, disc, Cs, gen_block_ids, disc_block_ids, is_detach=True).detach()
    test_img_rec = torch.clamp(_tanh_to_sigmoid(test_img_rec), 0.0, 1.0).detach()
    test_img_rec = torch.from_numpy(resize_images(test_img_rec.detach().cpu().numpy(), resize=256))
    save_img = torch.cat([test_img_reshape, test_img_rec], dim=3).cpu()
    torchvision.utils.save_image(save_img, file_name, nrow=4, padding=0)


def report_interpolation_image2image(images1, images2, n_interps, n_steps_per_interp, gen, disc, Cs, gen_block_ids,
                                     disc_block_ids):
    f_gen1 = compose_discriminator_forward(images1, disc, disc_block_ids, Cs, disc_detach=True, return_f_disc=False)
    f_gen2 = compose_discriminator_forward(images2, disc, disc_block_ids, Cs, disc_detach=True, return_f_disc=False)
    save_images = [post_process_image(images1, resize=256)]
    f_before = f_gen1
    for i in range(n_interps - 1):
        alpha = i / (n_interps - 1)
        interp_image = compose_forward(f_before, gen_block_ids, gen, return_last=True, if_detach=True)
        save_images.append(post_process_image(interp_image, resize=256))
        for j in range(n_steps_per_interp - 1):
            delta_alpha = 1 / ((n_interps - 1) * (n_steps_per_interp - 1))
            alpha_new = alpha + delta_alpha * j / (n_steps_per_interp - 1)
            f_base1 = f_gen1[0]
            f_base2 = f_gen2[0]
            # interpolate the base
            f_base_mix = alpha_new * f_base2 + (1 - alpha_new) * f_base1
            f_before[0] = f_base_mix
            for k in range(len(gen_block_ids) - 1):
                new_image = compose_forward(f_before, gen_block_ids, gen, return_last=True, if_detach=True)
                new_features = compose_discriminator_forward(new_image, disc, disc_block_ids, Cs, disc_detach=True,
                                                             return_f_disc=False)
                f_before[k+1] = new_features[k+1]

    interp_image = compose_forward(f_before, gen_block_ids, gen, return_last=True, if_detach=True)
    save_images.append(post_process_image(interp_image, resize=256))
    save_images.append(post_process_image(images2, resize=256))
    save_images = torch.cat(save_images, dim=3)
    return save_images


def report_interpolation_image2image_version2(images1, images2, n_interps, n_steps_per_interp, gen, disc, Cs, gen_block_ids,
                                     disc_block_ids):
    f_gen1 = compose_discriminator_forward(images1, disc, disc_block_ids, Cs, disc_detach=True, return_f_disc=False)
    f_gen2 = compose_discriminator_forward(images2, disc, disc_block_ids, Cs, disc_detach=True, return_f_disc=False)
    save_images = [post_process_image(images1, resize=256)]
    for i in range(n_interps - 1):
        alpha = i / (n_interps - 1)
        f_base_mixs = []
        for j in range(len(f_gen1)):
            f_base_mixs.append(alpha * f_gen2[j] + (1 - alpha) * f_gen1[j])
        interp_image_A = compose_forward(f_base_mixs, gen_block_ids, gen, return_last=True, if_detach=True)
        new_features = compose_discriminator_forward(interp_image_A, disc, disc_block_ids, Cs, disc_detach=True,
                                                     return_f_disc=False)
        for j in range(len(f_gen1)):
            if j == 0:
                pass
            else:
                f_base_mixs[j] = f_base_mixs[j] + min(alpha, 1 - alpha) * new_features[j]
        interp_image_B = compose_forward(f_base_mixs, gen_block_ids, gen, return_last=True, if_detach=True)
        save_images.append(post_process_image(interp_image_B, resize=256))

    save_images.append(post_process_image(images2, resize=256))
    save_images = torch.cat(save_images, dim=3)
    return save_images


def report_interpolation_image2image_version3(images1, images2, n_interps, gen, disc, Cs, gen_block_ids,
                                              disc_block_ids, resize=256):
    save_images = [post_process_image(images1, resize=resize)]

    for i in range(n_interps - 1):
        alpha = i / (n_interps - 1)
        images_mix = (1 - alpha) * images1 + alpha * images2
        f_mix = compose_discriminator_forward(images_mix, disc, disc_block_ids, Cs, disc_detach=True, return_f_disc=False)
        f_mix_images = compose_forward(f_mix, gen_block_ids, gen, return_last=True, if_detach=True)
        save_images.append(post_process_image(f_mix_images, resize=resize))

    save_images.append(post_process_image(images2, resize=resize))
    save_images = torch.cat(save_images, dim=3)
    return save_images


def report_interpolation_prior2image():
    pass


def report_interpolation_prior2prior():
    pass


def easy_forward(input_img, gen, disc, Cs, gen_block_ids, disc_block_ids, is_detach=False, get_gf=False):
    f_disc, attain_lists = disc([input_img], which_block=disc_block_ids[0], pre_model=True, attain_list=disc_block_ids[1:])

    f_all = [f_disc] + attain_lists[::-1]

    f_gen_recs = []
    for i in range(len(Cs)):
        if i == 0:
            f_gen_recs.append(pixel_norm(Cs[i](f_all[i])))
        else:
            f_gen_recs.append(Cs[i](f_all[i]))

    start = gen_block_ids[0]
    input_ = f_gen_recs[0]
    for i in range(1, len(gen_block_ids), 1):
        end = gen_block_ids[i]
        input_ = pixel_norm(gen([input_], free_model=True, start=start, end=end))
        input_ = pixel_norm(input_ + f_gen_recs[i])
        start = end
    x_rec = gen([input_], which_block=gen_block_ids[-1], post_model=True).detach()
    return x_rec


def batch_run(batch_size, input_img, run_func, *args, **kwargs):
    n = input_img.shape[0]
    results = []
    for i in range(0, n, batch_size):
        img = input_img[i:min(n, i+batch_size)]
        img_rec = run_func(img, *args, **kwargs)
        results.append(img_rec)
    return torch.cat(results, dim=0)


def compose_forward(f_gen_recs, gen_block_ids, gen, return_hidden=False, return_last=True, if_detach=False):
    input_ = f_gen_recs[0]
    if if_detach:
        input_ = input_.detach()

    start = gen_block_ids[0]
    hiddens = []
    if return_hidden:
        hiddens.append(input_)

    for i in range(1, len(gen_block_ids), 1):
        end = gen_block_ids[i]
        input_ = pixel_norm(gen([input_], which_block=None, free_model=True, start=start, end=end))
        input_ = pixel_norm(input_ + f_gen_recs[i])
        if if_detach:
            input_ = input_.detach()
        if return_hidden:
            hiddens.append(input_)
        start = end

    if return_last:
        x_gen = gen([input_], which_block=gen_block_ids[-1], post_model=True)
        if if_detach:
            x_gen = x_gen.detach()
        if return_hidden:
            return x_gen, hiddens
        else:
            return x_gen
    else:
        return hiddens


def compose_forward_v2(f_gen_recs, gen_block_ids, gen, return_hidden=False, return_last=True, if_detach=False):
    input_ = f_gen_recs[0]
    if if_detach:
        input_ = input_.detach()

    start = gen_block_ids[0]
    hiddens = []
    if return_hidden:
        hiddens.append(input_)

    for i in range(1, len(gen_block_ids), 1):
        end = gen_block_ids[i]
        input_ = pixel_norm(gen([input_], which_block=None, free_model=True, start=start, end=end))
        input_ = pixel_norm(input_ + f_gen_recs[i])
        if if_detach:
            input_ = input_.detach()
        if return_hidden:
            hiddens.append(input_)
        start = end

    if return_last:
        x_gen = gen([input_], which_block=gen_block_ids[-1], post_model=True)
        if if_detach:
            x_gen = x_gen.detach()
        if return_hidden:
            return x_gen, hiddens
        else:
            return x_gen
    else:
        return hiddens


# losses
def compute_loss(losses, layers, wgt_base, wgt_final, power=1.0, n_step=None):
    n_losses = len(losses)
    weights = []
    for i in range(n_losses):
        alpha = 1 - (layers[i] - layers[0])/(layers[-1] - layers[0])
        wgt = wgt_base * np.power(alpha, power) + np.power((1 - alpha), power) * wgt_final
        weights.append(wgt)

    loss = 0
    individual = []
    if n_step is None:
        n_step = n_losses

    for i in range(n_losses):
        individual.append(weights[i] * losses[i])
    for i in range(n_step):
        loss = loss + weights[i] * losses[i]
    return loss, individual




