import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .gan_utils import get_gan_model


PGGAN_LATENT_1024 = [(512, 1, 1),
              (512, 4, 4), (512, 4, 4),
              (512, 8, 8), (512, 8, 8),
              (512, 16, 16), (512, 16, 16),
              (512, 32, 32), (512, 32, 32),
              (256, 64, 64), (256, 64, 64),
              (128, 128, 128), (128, 128, 128),
              (64, 256, 256), (64, 256, 256),
              (32, 512, 512), (32, 512, 512),
              (16, 1024, 1024), (16, 1024, 1024),
              (3, 1024, 1024)]

PGGAN_LATENT_256 = [(512, 1, 1),
                    (512, 4, 4), (512, 4, 4),
                    (512, 8, 8), (512, 8, 8),
                    (512, 16, 16), (512, 16, 16),
                    (512, 32, 32), (512, 32, 32),
                    (256, 64, 64), (256, 64, 64),
                    (128, 128, 128), (128, 128, 128),
                    (64, 256, 256), (64, 256, 256),
                    (3, 256, 256)]

PGGAN_LAYER_MAPPING = {  # The new PGGAN includes the intermediate output layer, need mapping
    0: 0, 1: 1, 2: 3, 3: 4, 4: 6, 5: 7, 6: 9, 7: 10, 8: 12
}


def get_derivable_generator(gan_model_name, generator_type, args):
    if generator_type == 'PGGAN-z':  # Single latent code
        return PGGAN(gan_model_name)
    elif generator_type == 'BigGANDeep':  # Single latent code
        return BigGANDeep(gan_model_name)
    elif generator_type == 'BigGANShallow':
        return BigGANShallow(gan_model_name)
    elif generator_type == 'StyleGAN-z':
        return StyleGAN(gan_model_name, 'z')
    elif generator_type == 'StyleGAN-w':
        return StyleGAN(gan_model_name, 'w')
    elif generator_type == 'StyleGAN-w+':
        return StyleGAN(gan_model_name, 'w+')
    elif generator_type == 'StyleGAN-Layerwise-z':
        return StyleGAN_Layerwise(gan_model_name, 'z')
    elif generator_type == 'StyleGAN-Layerwise-w':
        return StyleGAN_Layerwise(gan_model_name, 'w')
    elif generator_type == 'StyleGAN-Layerwise-w+':
        return StyleGAN_Layerwise(gan_model_name, 'w+')
    elif generator_type == 'PGGAN-Multi-Z':  # Multiple Latent Codes
        return PGGAN_multi_z(gan_model_name, args.composing_layer, args.z_number, args)
    elif generator_type == 'PGGAN-Layerwise':
        return PGGAN_Layerwise(gan_model_name)
    elif generator_type == 'SNGAN':
        return SNGAN(gan_model_name)
    else:
        raise Exception('Please indicate valid `generator_type`')


class PGGAN(nn.Module):
    def __init__(self, gan_model_name):
        super(PGGAN, self).__init__()
        self.pggan = get_gan_model(gan_model_name)
        self.init = False

    def input_size(self):
        return [(512,)]

    def init_value(self, batch_size):
        z_estimate = torch.randn((batch_size, 512)).cuda()  # our estimate, initialized randomly
        return [z_estimate]

    def init_delta(self, batch_size):
        z_delta = torch.zeros((batch_size, 512)).cuda()  # our estimate, initialized randomly
        return [z_delta]

    def cuda(self, device=None):
        self.pggan.cuda(device=device)

    def forward(self, z):
        latent = z[0]
        latent = latent.view(-1, 512)
        return self.pggan(latent.view(latent.size(0), 512, 1, 1))


class StyleGAN(nn.Module):
    def __init__(self, gan_model_name, start):
        super(StyleGAN, self).__init__()
        self.stylegan = get_gan_model(gan_model_name).net
        self.start = start
        self.init = False

    def input_size(self):
        if self.start == 'z' or self.start == 'w':
            return [(512,)]
        elif self.start == 'w+':
            return [(self.stylegan.net.synthesis.num_layers, 512)]

    def cuda(self, device=None):
        self.stylegan.cuda(device=device)

    def forward(self, latent):
        z = latent[0]
        if self.start == 'z':
            w = self.stylegan.mapping(z)
            w = self.stylegan.truncation(w)
            x = self.stylegan.synthesis(w)
            return x
        elif self.start == 'w':
            z = z.view((-1, self.stylegan.net.synthesis.num_layers, self.stylegan.net.synthesis.w_space_dim))
            x = self.stylegan.net.synthesis(z)
            return x
        elif self.start == 'w+':
            x = self.stylegan.net.synthesis(z)
            return x


class StyleGAN_Layerwise(nn.Module):
    def __init__(self, gan_model_name, start):
        super(StyleGAN_Layerwise, self).__init__()
        self.stylegan = get_gan_model(gan_model_name).net
        self.start = start
        self.init = False
        self.n_layers = self.stylegan.mapping.num_layers + 1 + self.stylegan.synthesis.num_layers

    def input_size(self):
        if self.start == 'z' or self.start == 'w':
            return [(512,)]
        elif self.start == 'w+':
            return [(self.stylegan.net.synthesis.num_layers, 512)]

    def cuda(self, device=None):
        self.stylegan.cuda(device=device)

    def forward(self, latent, which_block=0, pre_model=False, post_model=False):
        if pre_model:
            z = latent[0]
            b, c, w, h = z.shape
            assert w == 1 and h == 1
            if self.start == 'z':
                z = z.view(b, c)
                x = self.stylegan(z, which_block=which_block, pre_model=pre_model, post_model=post_model)
                return x
            elif self.start == 'w':
                z = z.view((-1, self.stylegan.net.synthesis.num_layers, self.stylegan.net.synthesis.w_space_dim))
                x = self.stylegan.net.synthesis(z)
                return x
            elif self.start == 'w+':
                x = self.stylegan.net.synthesis(z)
                return x

        elif post_model:
            z = latent[0]
            if self.start == 'z':
                x = self.stylegan(z, which_block=which_block, pre_model=pre_model, post_model=post_model)
                return x
            elif self.start == 'w':
                z = z.view((-1, self.stylegan.net.synthesis.num_layers, self.stylegan.net.synthesis.w_space_dim))
                x = self.stylegan.net.synthesis(z)
                return x
            elif self.start == 'w+':
                x = self.stylegan.net.synthesis(z)
                return x

        else:
            z = latent[0]
            if self.start == 'z':
                w = self.stylegan.mapping(z)
                w = self.stylegan.truncation(w)
                x = self.stylegan.synthesis(w)
                return x
            elif self.start == 'w':
                z = z.view((-1, self.stylegan.net.synthesis.num_layers, self.stylegan.net.synthesis.w_space_dim))
                x = self.stylegan.net.synthesis(z)
                return x
            elif self.start == 'w+':
                x = self.stylegan.net.synthesis(z)
                return x


class PGGAN_multi_z(nn.Module):
    def __init__(self, gan_model_name, blending_layer, z_number, args):
        super(PGGAN_multi_z, self).__init__()
        self.blending_layer = blending_layer
        self.z_number = z_number
        self.z_dim = 512
        self.pggan = get_gan_model(gan_model_name)

        self.pre_model = nn.Sequential(*list(self.pggan.children())[:blending_layer])
        self.post_model = nn.Sequential(*list(self.pggan.children())[blending_layer:])
        self.init = True

        PGGAN_LATENT = PGGAN_LATENT_1024 if gan_model_name == 'PGGAN-CelebA' \
            else PGGAN_LATENT_256
        self.mask_size = PGGAN_LATENT[blending_layer][1:]
        self.layer_c_number = PGGAN_LATENT[blending_layer][0]

    def input_size(self):
        return [(self.z_number, self.z_dim), (self.z_number, self.layer_c_number)]

    def hidden_size(self):
        return self.mask_size, self.layer_c_number

    def init_value(self, batch_size):
        z_estimate = torch.randn((batch_size, self.z_number, self.z_dim)).cuda()  # our estimate, initialized randomly
        z_alpha = torch.full((batch_size, self.z_number, self.layer_c_number), 1 / self.z_number).cuda()
        return [z_estimate, z_alpha]

    def init_delta(self, batch_size):
        z_estimate = torch.zeros((batch_size, self.z_number, self.z_dim)).cuda()  # our estimate, initialized randomly
        # do not learn z.
        # z_alpha = torch.full((batch_size, self.z_number, self.layer_c_number), 0).cuda()
        return [z_estimate]

    def cuda(self, device=None):
        self.pggan.cuda(device=device)

    def forward(self, z, just_hidden=False, with_hidden=False, post_model=False, pre_model=False):
        if post_model:
            y_estimate = self.post_model(z)
            return y_estimate
        if pre_model:
            y_estimate = self.pre_model(z)
            return y_estimate
        if len(z) == 2:
            z_estimate, alpha_estimate = z
            feature_maps_list = []
            for j in range(self.z_number):
                feature_maps_list.append(
                    self.pre_model(z_estimate[:, j, :].view((-1, self.z_dim, 1, 1))) *
                    alpha_estimate[:, j, :].view((-1, self.layer_c_number, 1, 1)))

            fused_feature_map = sum(feature_maps_list)
            if just_hidden:
                return fused_feature_map
            else:
                y_estimate = self.post_model(fused_feature_map)
                if with_hidden:
                    return y_estimate, fused_feature_map
                else:
                    return y_estimate
        else:
            z_estimate = z[0]
            return self.pre_model(z_estimate.view((-1, self.z_dim, 1, 1)))


class SNGAN(nn.Module):
    # dim is 128
    def __init__(self, gan_model_name):
        super(SNGAN, self).__init__()
        self.net = get_gan_model(gan_model_name)
        self.sngan = self.net.net
        self.n_blocks = len(list(self.sngan.children())[0])
        print('There are %d blocks in sngan.' % self.n_blocks)
        self.init = False
        self.config = self.net.config

        for i in range(self.n_blocks):
            setattr(self, 'block_%d' % i, nn.Sequential(*[list(self.sngan.children())[0][i]]))
            setattr(self, 'post_model_%d' % i, nn.Sequential(*list(self.sngan.children())[0][i:]))
            setattr(self, 'pre_model_%d' % i, nn.Sequential(*list(self.sngan.children())[0][:i]))

    def cuda(self, device=None):
        self.sngan.cuda(device=device)

    def forward(self, x, which_block, post_model=False, pre_model=False):
        h = x[0]
        if pre_model:
            h = h.view(h.shape[0], h.shape[1])

        if post_model:
            h = getattr(self, "post_model_%d" % which_block)(h)
        elif pre_model:
            h = getattr(self, "pre_model_%d" % which_block)(h)
        else:
            h = getattr(self, 'block_%d' % which_block)(h)
        return h


class PGGAN_Layerwise(nn.Module):
    def __init__(self, gan_model_name):
        super(PGGAN_Layerwise, self).__init__()
        self.z_dim = 512
        self.pggan = get_gan_model(gan_model_name)
        self.n_blocks = len(list(self.pggan.children()))
        print("There are %d blocks in pggan." % self.n_blocks)
        self.init = True

        for i in range(self.n_blocks):
            setattr(self, 'block_%d' % i, nn.Sequential(*[list(self.pggan.children())[i]]))
            setattr(self, 'post_model_%d' % i, nn.Sequential(*list(self.pggan.children())[i:]))
            setattr(self, 'pre_model_%d' % i, nn.Sequential(*list(self.pggan.children())[:i]))

        PGGAN_LATENT = PGGAN_LATENT_1024 if gan_model_name == 'pggan_celebahq' \
            else PGGAN_LATENT_256
        self.PGGAN_LATENT = PGGAN_LATENT

    def forward(self, x, which_block, post_model=False, pre_model=False):
        h = x[0]
        if post_model:
            h = getattr(self, "post_model_%d" % which_block)(h)
        elif pre_model:
            h = getattr(self, "pre_model_%d" % which_block)(h)
        else:
            h = getattr(self, 'block_%d' % which_block)(h)
        return h

    def input_size(self, which_layer):
        return [self.PGGAN_LATENT[which_layer]]

    def init_value(self, batch_size, which_layer, init="Zeros", z_numbers=1):
        if init.upper() == "ZEROS":
            z_estimate = torch.zeros((z_numbers, ) + (batch_size,) + self.PGGAN_LATENT[which_layer]).cuda()  # our estimate, initialized randomly
        elif init.upper() == 'GAUSSIAN':
            z_estimate = torch.randn((z_numbers, ) + (batch_size,) + self.PGGAN_LATENT[which_layer]).cuda()
        else:
            raise NotImplemented('We dont have this initialization. ')

        return [z_estimate]

    def init_pvalue(self, batch_size,  which_layer, z_numbers=1, init="Uniform", div_factor=None):
        if init == "Zeros":
            p_estimate = torch.zeros((z_numbers, batch_size, self.PGGAN_LATENT[which_layer][0], 1, 1)).cuda()  # our estimate, initialized randomly
        elif init == "Uniform":
            if div_factor is None:
                p_estimate = torch.ones((z_numbers, batch_size, self.PGGAN_LATENT[which_layer][0], 1, 1)).cuda() / z_numbers
            else:
                p_estimate = torch.ones(
                    (z_numbers, batch_size, self.PGGAN_LATENT[which_layer][0], 1, 1)).cuda() / div_factor
        else:
            raise NotImplemented('We don\'t support this type of initialization.')

        return [p_estimate]

    def init_multi_pvalue(self, batch_size, per_layer, init="Uniform"):
        p_estimate = []
        for layer in range(1, len(self.PGGAN_LATENT)-1, per_layer):
            if init == 'Uniform':
                estimate = torch.ones((1, batch_size, self.PGGAN_LATENT[layer][0], 1, 1)).cuda() * 0.5
            else:
                raise NotImplemented('We dont have this type of initialization.')
            p_estimate.append(estimate)
        return p_estimate

    def init_bvalue(self, batch_size, which_layers, init='Zeros'):
        res = []
        if init == 'Zeros':
            for layer in which_layers:
                b_estimate = torch.zeros((batch_size, 1,) + self.PGGAN_LATENT[layer+1][1:]).cuda()   # our estimate, initialized randomly
                res += [b_estimate]
        else:
            for layer in which_layers:
                b_estimate = torch.randn((batch_size, 1,) + self.PGGAN_LATENT[layer+1][1:]).cuda()
                res += [b_estimate]
        return res

    def cuda(self, device=None):
        self.pggan.cuda(device=device)


class BigGANDeep(nn.Module):
    def __init__(self, gan_model_name):
        super(BigGANDeep, self).__init__()
        self.biggan = get_gan_model(gan_model_name)
        self.init = False
        self.config = self.biggan.config

    def input_size(self):
        return [(self.config.z_dim,), (self.config.num_classes,)]

    def init_value(self, batch_size):
        z_estimate = torch.randn((batch_size, self.config.z_dim)).cuda()  # our estimate, initialized randomly
        y_estimate = torch.randint(0, self.config.num_classes, size=(batch_size,), dtype=torch.int64).cuda()
        return [z_estimate, y_estimate]

    def cuda(self, device=None):
        self.biggan.cuda(device=device)

    def to_one_hot(self, label):
        if isinstance(label, np.ndarray):
            num = label.shape[0]
        elif isinstance(label, torch.Tensor):
            num = label.size(0)
        else:
            raise NotImplemented()
        n_classes = self.config.num_classes
        label_one_hot = np.zeros(shape=[num, n_classes], dtype=np.float32)
        for i in range(num):
            label_one_hot[i][label[i]] = 1.0

        if isinstance(label, np.ndarray):
            return label_one_hot
        elif isinstance(label, torch.Tensor):
            return torch.from_numpy(label_one_hot).cuda()
        else:
            raise NotImplemented()

    # Usage:
    # model = BigGANDeep('biggandeep128_imagenet')
    # z = torch.rand(size=(batch_size, 128), dtype=torch.float32).cuda()
    # class_label = get_random_int(55, 1000)
    # truncation = 1.0    # we don't use truncation.
    #   # fully forward:
    # x = model([z, class_label], truncation)
    #   # partial forward:
    # f = model([z, class_label], truncation, which_block=4, pre_model=True)
    # x = model([z, class_label], truncation, f, which_block=4, post_model=True)
    def forward(self, input_pack, truncation, features=None, which_block=50, pre_model=False, post_model=False):
        latent = input_pack[0]
        label = self.to_one_hot(input_pack[1])

        latent = latent.view(-1, self.config.z_dim)
        return self.biggan(z=latent.view(latent.size(0), self.config.z_dim),
                           class_label=label,
                           truncation=truncation,
                           features=features,
                           which_block=which_block,
                           pre_model=pre_model,
                           post_model=post_model)


class BigGANShallow(nn.Module):
    def __init__(self, gan_model_name):
        super(BigGANShallow, self).__init__()
        self.net = get_gan_model(gan_model_name)
        self.biggan = self.net.net
        self.init = False
        self.config = self.net.config

    def cuda(self, device=None):
        self.biggan.cuda(device=device)

    def forward(self, input_pack, *args):
        z, which_class = input_pack
        bs, c, w, h = z.shape
        assert w == 1 and h == 1
        z_reshape = z.view(bs, c)
        x = self.biggan(z_reshape, self.biggan.shared(which_class))
        return x



