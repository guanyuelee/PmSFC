import os
import argparse
import numpy as np
import h5py

import torch
from torch import nn

from derivable_models.derivable_generator import get_derivable_generator


def forward_discriminator(x_gen, disc, disc_block_ids, disc_detach):
    if len(disc_block_ids[1:]) == 0:
        f_disc = disc([x_gen], which_block=disc_block_ids[0], pre_model=True, attain_list=disc_block_ids[1:])
        f_attns = []
    else:
        f_disc, f_attns = disc([x_gen], which_block=disc_block_ids[0], pre_model=True, attain_list=disc_block_ids[1:])
    if disc_detach:
        f_disc_features = [f_disc.detach()] + [item.detach() for item in f_attns[::-1]]
    else:
        f_disc_features = [f_disc] + f_attns[::-1]
    return f_disc_features


def main(args):
    disc = nn.DataParallel(get_derivable_generator(args.disc_model, args.disc_type, args)).cuda()
    gen = nn.DataParallel(get_derivable_generator(args.gan_model, args.gan_type, args)).cuda()

    total_blocks = 18 if args.resolution == 1024 else 14
    gen_block_ids = list(np.linspace(1, total_blocks - 4, total_blocks - 4).astype(np.int).tolist())
    disc_block_ids = [gen.module.n_blocks - item for item in gen_block_ids]

    batch_size = args.batch_size
    num_data = args.num_data
    count = 0
    for i in range(0, num_data, batch_size):
        print('(%d/%d)' % (count, num_data))
        if i == 0:
            print('produce dataset of len %d. ' % num_data)
            with h5py.File(os.path.join(args.outputs, args.file_name), 'w') as file:
                z = torch.randn(size=(batch_size, args.dim_z), dtype=torch.float32).cuda()
                if num_data - count < batch_size:
                    z = z[:(num_data - count)]
                noise_dset = file.create_dataset('noise', z.shape, dtype=np.float32,
                                    maxshape=(num_data, args.dim_z), chunks=(args.chunk_size, args.dim_z),
                                    compression=args.compression)
                noise_dset[...] = z.cpu().detach().numpy()

                h = z
                for layer_i in range(len(gen_block_ids)):
                    layer_name = 'gen_layer_%d' % gen_block_ids[layer_i]
                    if layer_i == 0:
                        h = gen([z.reshape((batch_size, args.dim_z, 1, 1))], pre_model=True, which_block=gen_block_ids[layer_i]).detach()
                    else:
                        h = gen([h], free_model=True, start=gen_block_ids[layer_i - 1], end=gen_block_ids[layer_i]).detach()
                    h_numpy = h.detach().cpu().numpy()
                    layer_i_dset = file.create_dataset(layer_name,
                                                       maxshape=(num_data, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                       dtype=np.float32, chunks=(args.chunk_size, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                       compression=args.compression)
                    layer_i_dset[...] = h_numpy

                h = gen([h], post_model=True, which_block=gen_block_ids[-1]).detach()
                name = 'generation'
                h_numpy = h.detach().cpu().numpy()
                generation_dset = file.create_dataset(name,
                                                      maxshape=(num_data, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                      dtype=np.float32,
                                                      chunks=(args.chunk_size, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                      compression=args.compression)
                generation_dset[...] = h_numpy

                results = forward_discriminator(h, disc, disc_block_ids, disc_detach=True)

                for layer_i in range(len(results)):
                    name = 'disc_layer_%d' % disc_block_ids[layer_i]
                    h_numpy = results[layer_i].detach().cpu().numpy()
                    layer_i_dset = file.create_dataset(name,
                                                       maxshape=(num_data, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                       dtype=np.float32,
                                                       chunks=(args.chunk_size, h_numpy.shape[1], h_numpy.shape[2], h_numpy.shape[3]),
                                                       compression=args.compression)
                    layer_i_dset[...] = h_numpy
        else:
            with h5py.File(os.path.join(args.outputs, args.file_name), 'a') as file:
                h = torch.randn(size=(batch_size, args.dim_z), dtype=torch.float32).cuda()
                if num_data - count < batch_size:
                    h = h[:(num_data - count)]
                h_numpy = h.detach().cpu().numpy()
                file['noise'].resize(file['noise'].shape[0] + h_numpy.shape[0], axis=0)
                file['noise'][-h_numpy.shape[0]:] = h_numpy

                for layer_i in range(len(gen_block_ids)):
                    layer_name = 'gen_layer_%d' % gen_block_ids[layer_i]
                    if layer_i == 0:
                        h = gen([h.reshape((batch_size, args.dim_z, 1, 1))], pre_model=True, which_block=gen_block_ids[layer_i]).detach()
                    else:
                        h = gen([h], free_model=True, start=gen_block_ids[layer_i - 1], end=gen_block_ids[layer_i]).detach()
                    h_numpy = h.detach().cpu().numpy()
                    file[layer_name].resize(file[layer_name].shape[0] + h_numpy.shape[0], axis=0)
                    file[layer_name][-h_numpy.shape[0]:] = h_numpy

                h = gen([h], post_model=True, which_block=gen_block_ids[-1]).detach()
                name = 'generation'
                h_numpy = h.detach().cpu().numpy()
                file[name].resize(file[name].shape[0] + h_numpy.shape[0], axis=0)
                file[name][-h_numpy.shape[0]:] = h_numpy

                results = forward_discriminator(h, disc, disc_block_ids, disc_detach=True)

                for layer_i in range(len(results)):
                    name = 'disc_layer_%d' % disc_block_ids[layer_i]
                    h_numpy = results[layer_i].detach().cpu().numpy()
                    file[name].resize(file[name].shape[0] + h_numpy.shape[0], axis=0)
                    file[name][-h_numpy.shape[0]:] = h_numpy
        count += batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploit discriminator feature for superior representation.')
    parser.add_argument('--disc_model', type=str, default='pggan_celebahq_disc',
                        help='The name of the discriminator. ')
    parser.add_argument('--gan_model', type=str, default='pggan_celebahq',
                        help='The name of the generator. ')
    parser.add_argument('--disc_type', default='PGGAN-Disc-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    parser.add_argument('--gan_type', default='PGGAN-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    parser.add_argument('--resolution', default=1024, type=int,
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    parser.add_argument('--outputs', type=str, default='./datasets', help='The directory of output. ')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size of the image. ')
    parser.add_argument('--num_data', type=int, default=70000, help='The total iteration of training. ')
    parser.add_argument('--dim_z', type=int, default=512, help='The total iteration of training. ')

    parser.add_argument('--file_name', type=str, default='pggan_celebahq.h5', help='The total iteration of training. ')
    parser.add_argument('--chunk_size', type=int, default=500, help='Default overall batchsize (default: %(default)s)')
    parser.add_argument('--compression', action='store_true', default=False, help='Use LZF compression? (default: %(default)s)')
    # parser.add_argument('--gen_block_ids', type=str, default='4-8', help='The block which composition. ')

    args = parser.parse_args()
    main(args)
