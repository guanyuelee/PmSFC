import os
import argparse
import torch
import torch.nn as nn
import cv2
import copy
import pickle as pkl
import numpy as np

from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
from derivable_models.derivable_generator import get_derivable_generator
from models.model_settings import MODEL_POOL
from utils.file_utils import create_experiments_directory
import torch.optim as optim
import torchvision

from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize


# threshold the matrix.
def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t, i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def main(args):
    os.makedirs(args.outputs, exist_ok=True)
    out_dir, exp_name = create_experiments_directory(args, args.exp_id)
    print(out_dir)
    print(exp_name)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator.cuda()

    if args.target_images.endswith('.png') or args.target_images.endswith('.jpg'):
        image_list = os.path.abspath(args.target_images)
        image_list = [image_list]
    else:
        image_list = image_files(args.target_images)
    frameSize = MODEL_POOL[args.gan_model]['resolution']
    n_blocks = generator.n_blocks
    print('There are %d blocks in this generator.' % n_blocks)  # 19 for pggan

    latent_space = generator.PGGAN_LATENT[args.layer]
    print('The latent space is ', latent_space)

    with open(args.matrix_dir, 'rb') as file_in:
        matrix = pkl.load(file_in)
        print('Load matrix successfully.')
        print('Matrix shape ', matrix.shape)

    matrix2 = thrC(matrix, args.alpha)
    predict, _ = post_proC(matrix2, args.cluster_numbers, args.subspace_dimension, args.power)
    print(predict)
    p_sum = [sum(predict == k) for k in range(1, args.cluster_numbers, 1)]
    p_sum = np.array(p_sum)
    p_sort = np.argsort(p_sum)[::-1]
    print(p_sum)
    predict_new = predict.copy()
    for i in range(1, args.cluster_numbers, 1):
        predict_new[predict == (p_sort[i - 1]+1)] = i
    predict = predict_new.copy()
    p_sum = [sum(predict == k) for k in range(1, args.cluster_numbers, 1)]
    print(predict)
    print(p_sum)

    # pre_see the images
    gan_type, image_type = args.gan_model.split("_")
    print('The gan type is %s, and the image type is %s' % (gan_type, image_type))
    test_image_dir = os.path.join('./bin/', gan_type, image_type)
    print(test_image_dir)

    files = os.listdir(test_image_dir)
    test_zs = []
    for i in range(len(files)):
        if files[i].endswith('.pkl'):
            with open(os.path.join(test_image_dir, files[i]), 'rb') as file_in:
                test_zs.append(pkl.load(file_in))
    test_zs = torch.from_numpy(np.concatenate(test_zs, axis=0).astype(np.float32)).cuda()
    print('Load all testing zs, shape is ', test_zs.size())

    image_number = 3
    sel_idx = np.random.choice(test_zs.shape[0], size=[image_number], replace=False)
    F = generator([test_zs[sel_idx]], which_block=args.layer, pre_model=True)
    features = F.detach().cpu().numpy()

    '''
    for class_i in range(1, args.cluster_numbers + 1, 1):
        print(class_i)
        ex_images = []
        for ii in range(image_number):
            ex_rows = []
            for jj in range(image_number):
                f_a = features[ii].copy()
                f_b = features[jj].copy()
                f_a[predict == class_i] = f_b[predict == class_i]
                f_a = f_a.reshape((1,) + latent_space)
                ex_rows.append(f_a)
            ex_rows = np.concatenate(ex_rows, axis=0).astype(np.float32)
            ex_rows = torch.from_numpy(ex_rows).cuda()
            ex_ys = generator([ex_rows], which_block=args.layer, post_model=True).detach()
            ex_ys = torch.clamp(_tanh_to_sigmoid(ex_ys), min=0.0, max=1.0)
            ex_images.append(ex_ys)
        ex_images = torch.cat(ex_images, dim=0)
        torchvision.utils.save_image(ex_images.detach().cpu(),
                                     os.path.join(out_dir, 'pre_see_class_%d.png' % (class_i)),
                                     nrow=image_number)
    '''

    predict_masks = []
    for i in range(1, args.cluster_numbers + 1, 1):
        mask = torch.from_numpy((predict == i).astype(np.float32)).cuda()
        predict_masks += [mask.reshape((1, -1, 1, 1))]

    for i, images in enumerate(split_to_batches(image_list, 1)):
        # input_size = generator.PGGAN_LATENT[args.layer + 1]
        # print("We are making ", input_size, ". ")
        print('%d: Inverting %d images :' % (i + 1, 1), end='')
        pt_image_str = '%s\n'
        print(pt_image_str % tuple(images))
        image_name_only = images[0].split(".")[0]
        image_name_only = image_name_only.split("/")[-1]
        print(image_name_only)

        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            image_tensor_list.append(_add_batch_one(load_as_tensor(image)))
        print("image_name_list", image_name_list)
        print("image_tensor_list, [", image_tensor_list[0].size(), "]")

        y_image = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda()
        print('image size is ', y_image.size())

        z_estimate = generator.init_value(batch_size=1, which_layer=0,
                                          init=args.init_type,
                                          z_numbers=args.cluster_numbers * args.code_per_cluster)
        base_estimate = generator.init_value(batch_size=1, which_layer=0,
                                          init=args.init_type,
                                          z_numbers=1)

        if args.optimization == 'GD':
            z_optimizer = torch.optim.SGD(z_estimate + base_estimate, lr=args.lr)
        elif args.optimization == 'Adam':
            z_optimizer = torch.optim.Adam(z_estimate + base_estimate, lr=args.lr)
        else:
            raise NotImplemented('We don\'t support this type of optimization.')

        for iter in range(args.iterations):
            for estimate in z_estimate:
                estimate.requires_grad = True
            for estimate in base_estimate:
                estimate.requires_grad = True

            features = generator([z_estimate[0].reshape([args.cluster_numbers * args.code_per_cluster, 512, 1, 1])],
                                 which_block=args.layer, pre_model=True)
            base_feature = generator([base_estimate[0].reshape([1, 512, 1, 1])], which_block=args.layer, pre_model=True)

            for t in range(args.cluster_numbers * args.code_per_cluster):
                if t == 0:
                    f_mix = features[t].view(*(1, )+latent_space) * predict_masks[int(t / args.code_per_cluster)]
                else:
                    f_mix = f_mix + features[t].view(*(1, )+latent_space) * predict_masks[int(t / args.code_per_cluster)]

            f_mix = f_mix + base_feature
            y_estimate = generator([f_mix], which_block=args.layer, post_model=True)
            y_raw_estimate = generator([base_feature], which_block=args.layer, post_model=True)
            z_optimizer.zero_grad()
            loss = 0.01 * torch.mean(torch.pow(y_estimate - y_image, 2.0)) + torch.mean(torch.pow(y_raw_estimate - y_image, 2.0))
            loss.backward()
            z_optimizer.step()

            if iter % args.report_value == 0:
                print('Iter %d, layer %d, loss = %.4f.' % (iter, args.layer, float(loss.item())))

            if iter % args.report_image == 0:
                print('Saving the images.')
                y_estimate_pil = Tensor2PIL(
                    torch.clamp(_tanh_to_sigmoid(y_estimate.detach().cpu()), min=0.0, max=1.0))
                y_estimate_pil.save(os.path.join(out_dir, image_name_only + "_estimate_iter%d.png" % iter))
                y_estimate_pil = Tensor2PIL(
                    torch.clamp(_tanh_to_sigmoid(y_raw_estimate.detach().cpu()), min=0.0, max=1.0))
                y_estimate_pil.save(os.path.join(out_dir, image_name_only + "_raw_estimate_iter%d.png" % iter))
                # add bias added output picture.
                # save all the codes
                codes = []
                for code_idx in range(args.cluster_numbers * args.code_per_cluster):
                    code_f = generator([z_estimate[0][code_idx]], which_block=args.layer + 1, pre_model=True)
                    code_y = generator([code_f], which_block=args.layer + 1, post_model=True).detach().cpu()
                    codes.append(torch.clamp(_tanh_to_sigmoid(code_y), min=0, max=1))
                codes = torch.cat(codes, dim=0).detach().cpu()
                torchvision.utils.save_image(codes, os.path.join(out_dir,
                                                                 image_name_only + '_codes_iter%d.png' % (
                                                                     iter)),
                                             nrow=(args.cluster_numbers * args.code_per_cluster) // 2)

            if iter % args.report_model == 0:
                print('Save the models')
                save_dict = {"z": z_estimate[0].detach().cpu().numpy(),
                             'matrix': matrix,
                             'layer': args.layer,
                             'predict': predict}
                with open(os.path.join(out_dir,
                                       'save_dict_iter_%d_layer_%d.pkl' % (iter, args.layer)), 'wb') as file_out:
                    pkl.dump(save_dict, file_out)
                print('Save the models OK!')


def str2bool(s):
    if s.lower() in ['yes', 'true', 'y', 't']:
        return True
    elif s.lower() in ['no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplemented()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer-Wise Inversion')
    # Image Path and Saving Path
    parser.add_argument('-i', '--target_images',
                        default='./examples/face',
                        help='Target images to invert.')
    parser.add_argument('-o', '--outputs',
                        default='./TRAIN',
                        help='Path to save results.')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='pggan_celebahq',
                        help='The name of model used.', type=str)
    parser.add_argument('--report_image', type=int, default=500)
    parser.add_argument('--report_value', type=int, default=10)
    parser.add_argument('--report_model', type=int, default=500)
    # Loss Parameters
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Size of images for perceptual model')
    parser.add_argument('--optimization', default='Adam',
                        help="['GD', 'Adam']. Optimization method used.")
    parser.add_argument('--init_type', default='Zeros',
                        help="['Zero', 'Normal']. Initialization method. Using zero init or Gaussian random vector.")
    parser.add_argument('--lr', default=1e-4,
                        help='Learning rate.', type=float)
    parser.add_argument('--p_lr', default=1e-2,
                        help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=50000,
                        help='Number of optimization steps.', type=int)
    parser.add_argument('--batch_size', default=1,
                        type=int, help='The number of batchsize')
    parser.add_argument('--layer', default=3,
                        type=int, help='The number of batchsize')
    parser.add_argument('--exp_id', default="SelfExpressInversion",
                        type=str, help='The number of batchsize')
    parser.add_argument('--matrix_dir', help='The path of matrix directory.',
                        type=str, default='./bin/pggan/celebahq/good_matrix/value1800.pkl')
    parser.add_argument('--code_per_cluster', help='Number of code per cluster',
                        default=1, type=int)
    parser.add_argument('--cluster_numbers', type=int, default=6, help='The number of cluster')
    parser.add_argument('--subspace_dimension', type=int, default=6, help='The number of subspace dimension.')
    parser.add_argument('--power', type=float, default=3.0, help='The power of the alpha.')
    parser.add_argument('--alpha', type=float, default=0.2, help='The power of the alpha.')

    # Video Settings
    parser.add_argument('--video', type=str2bool, default=False, help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate of the created video.')
    args, other_args = parser.parse_known_args()

    ### RUN
    main(args)
