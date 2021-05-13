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
import matplotlib.pyplot as plt


# threshold the matrix.
def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
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


def switch_matrix(C, predict):
    n = C.shape[0]
    n_classes = max(predict)
    class_counter = [sum(predict == (1 + i)) for i in range(n_classes)]
    print(class_counter)
    class_counter = list(np.cumsum(class_counter))
    print(class_counter)
    class_counter = [0] + class_counter
    print(class_counter)
    tmp_counter = [0 for i in range(n_classes)]

    def exchange_element(slice, i, j):
        kk = slice[i].copy()
        slice[i] = slice[j].copy()
        slice[j] = kk
        return slice.copy()

    for i in range(n):
        class_id = predict[i] - 1
        exchange_idx = class_counter[class_id] + tmp_counter[class_id]
        # exchange i and exchange_idx
        row_i = exchange_element(C[i, :].copy(), i, exchange_idx)
        col_i = exchange_element(C[:, i].copy(), i, exchange_idx)
        row_j = exchange_element(C[exchange_idx, :].copy(), i, exchange_idx)
        col_j = exchange_element(C[:, exchange_idx].copy(), i, exchange_idx)
        C[i, :] = row_j
        C[:, i] = col_j
        C[exchange_idx, :] = row_i
        C[:, exchange_idx] = col_i
        tmp_counter[class_id] += 1

    return C.copy()


def main(args):
    os.makedirs(args.outputs, exist_ok=True)
    out_dir, exp_name = create_experiments_directory(args, args.exp_id)
    print(out_dir)
    print(exp_name)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    generator.cuda()

    print('There are %d blocks in this generator.' % generator.n_blocks)  # 19 for pggan
    latent_space = generator.PGGAN_LATENT[args.layer]
    print('The latent space is ', latent_space)

    Matrix = torch.ones([latent_space[0], latent_space[0]], dtype=torch.float32).cuda() * 0.0001
    Matrix.requires_grad = True

    gan_type, image_type = args.gan_model.split("_")
    print('The gan type is %s, and the image type is %s' % (gan_type, image_type))
    test_image_dir = os.path.join('./bin/', gan_type, image_type)
    print(test_image_dir)

    if args.optimization == 'GD':
        optimizer = torch.optim.SGD([Matrix], lr=args.lr)
    elif args.optimization == 'Adam':
        optimizer = torch.optim.Adam([Matrix], lr=args.lr)
    else:
        raise NotImplemented('Not Implemented.')

    files = os.listdir(test_image_dir)
    test_zs = []
    for i in range(len(files)):
        if files[i].endswith('.pkl'):
            with open(os.path.join(test_image_dir, files[i]), 'rb') as file_in:
                test_zs.append(pkl.load(file_in))
    test_zs = torch.from_numpy(np.concatenate(test_zs, axis=0).astype(np.float32)).cuda()
    print(test_zs.size())

    losses_dict = {'total_loss': [], 'feature_loss': [], 'data_loss': [], 'sparse_loss': []}

    for iter in range(args.iterations):
        if args.beta0 > 0:
            Z = torch.randn([args.batch_size, 512, 1, 1], dtype=torch.float32).cuda()
            F = generator([Z], which_block=args.layer, pre_model=True)
            F_reshape = F.transpose(0, 1).reshape((latent_space[0], -1))
            F_rec = torch.matmul(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda()), F_reshape).\
                reshape((latent_space[0], args.batch_size,) + tuple(latent_space[1:])).transpose(0, 1)
            X_rec = generator([F_rec], which_block=args.layer, post_model=True)
            X = generator([F], which_block=args.layer, post_model=True)

            optimizer.zero_grad()
            if args.sparse_type == 'L2':
                sparse_loss = torch.mean(torch.pow(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda()), 2.0))
            elif args.sparse_type == 'L1':
                sparse_loss = torch.mean(torch.abs(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda())))
            else:
                raise NotImplemented('Type not implemented.')

            feature_loss = torch.mean(torch.pow(F - F_rec, 2.0))
            data_loss = torch.mean(torch.pow(X - X_rec, 2.0))
            loss = feature_loss + args.beta0 * data_loss + args.beta1 * sparse_loss
            loss.backward()
            optimizer.step()

        elif args.beta0 == 0:
            Z = torch.randn([args.batch_size, 512, 1, 1], dtype=torch.float32).cuda()
            F = generator([Z], which_block=args.layer, pre_model=True)
            F_reshape = F.transpose(0, 1).reshape((latent_space[0], -1))
            F_rec = torch.matmul(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda()), F_reshape). \
                reshape((latent_space[0], args.batch_size,) + tuple(latent_space[1:])).transpose(0, 1)

            optimizer.zero_grad()
            if args.sparse_type == 'L2':
                sparse_loss = torch.mean(
                    torch.pow(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda()), 2.0))
            elif args.sparse_type == 'L1':
                sparse_loss = torch.mean(
                    torch.abs(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda())))
            else:
                raise NotImplemented('Type not implemented.')
            feature_loss = torch.mean(torch.pow(F - F_rec, 2.0))
            loss = feature_loss + args.beta1 * sparse_loss
            loss.backward()
            optimizer.step()
        else:
            raise NotImplemented()

        if iter % args.report_value == 0:
            if args.beta0 == 0:
                print('Iter %d, Layer %d, loss=%.6f, f_loss=%.6f, sparse_loss=%.6f' %
                      (iter, args.layer, float(loss.item()), float(feature_loss.item()), float(sparse_loss.item())))
            else:
                print('Iter %d, Layer %d, loss=%.6f, f_loss=%.6f, x_loss=%.6f, sparse_loss=%.6f' %
                      (iter, args.layer, float(loss.item()), float(feature_loss.item()),
                       float(data_loss.item()), float(sparse_loss.item())))
                losses_dict['total_loss'].append(float(loss.item()))
                losses_dict['feature_loss'].append(float(feature_loss.item()))
                losses_dict['data_loss'].append(float(data_loss.item()))
                losses_dict['sparse_loss'].append(float(sparse_loss.item()))

        if iter % args.report_image == 0:
            # save reconstruction images.
            if args.beta0 == 0:
                X_rec = generator([F_rec], which_block=args.layer, post_model=True).detach()
                X = generator([F], which_block=args.layer, post_model=True).detach()

            test_feature = generator([test_zs[:4]], which_block=args.layer, pre_model=True).detach()
            test_feature_reshape = test_feature.transpose(0, 1).reshape((latent_space[0], -1))
            test_feature_rec = torch.matmul(Matrix * (1. - torch.eye(n=latent_space[0], m=latent_space[0]).cuda()),
                                            test_feature_reshape). \
                reshape((latent_space[0], args.batch_size,) + tuple(latent_space[1:])).transpose(0, 1)
            test_rec = generator([test_feature_rec], which_block=args.layer, post_model=True).detach()
            test_images = generator([test_feature], which_block=args.layer, post_model=True).detach()
            image_number = min(64, args.batch_size)
            Xs = torch.clamp(_tanh_to_sigmoid(torch.cat([test_rec,
                                                         test_images], dim=0).detach().cpu()), min=0.0, max=1.0)
            torchvision.utils.save_image(Xs, os.path.join(out_dir, 'ReconstructionImages_%d.png' % iter), nrow=4)

            # visualize the affinity matrix.
            Matrix_abs = torch.relu(Matrix)
            S_val = Matrix_abs.detach().cpu().numpy()
            S_val = thrC(S_val.copy().T, args.alpha).T

            #S_val_norm = S_val / np.sum(S_val, axis=1, keepdims=True)
            #S_val_norm = S_val_norm / np.sum(S_val_norm, axis=0, keepdims=True)
            #S_val_visual = np.clip(S_val_norm, 0, args.times * 1/512)
            #S_val_visual = S_val_visual / np.max(S_val_visual, axis=1, keepdims=True)
            #plt.figure()
            #plt.imshow(S_val_visual, cmap='Oranges')
            #plt.savefig(os.path.join(out_dir, 'thrd_matrix_iter_%d.png' % iter))

            predict, L_val = post_proC(S_val, args.n_subs, args.d_subs, args.power)

            p_sum = [sum(predict == k) for k in range(1, args.n_subs+1, 1)]
            p_sum = np.array(p_sum)
            print(p_sum)
            p_sort = np.argsort(p_sum)[::-1]
            predict_new = predict.copy()
            for i in range(1, args.n_subs+1, 1):
                predict_new[predict == (p_sort[i - 1] + 1)] = i
            predict = predict_new.copy()
            p_sum = [sum(predict == k) for k in range(1, args.n_subs+1, 1)]
            print(predict)
            print(p_sum)

            S_val_blockized = switch_matrix(S_val.copy(), predict.copy())
            # S_val_blockized = np.clip(S_val_blockized, 0, S_val_blockized.mean() * args.times)
            torchvision.utils.save_image(torch.from_numpy(S_val_blockized + S_val_blockized.T),
                                         os.path.join(out_dir, 'SwitchedMatrix%d.png' % iter), nrow=1,
                                         normalize=True)
            sel_idx = np.random.choice(test_zs.shape[0], size=[image_number], replace=False)
            F = generator([test_zs[sel_idx]], which_block=args.layer, pre_model=True).detach()
            features = F.detach().cpu().numpy()

            for class_i in range(1, args.n_subs+1, 1):
                ex_images = []
                for ii in range(image_number):
                    ex_rows = []
                    for jj in range(image_number):
                        f_a = features[ii].copy()
                        f_b = features[jj].copy()
                        f_a[predict == class_i] = f_b[predict == class_i]
                        f_a = f_a.reshape((1, ) + latent_space)
                        ex_rows.append(f_a)
                    ex_rows = np.concatenate(ex_rows, axis=0).astype(np.float32)
                    ex_rows = torch.from_numpy(ex_rows).cuda()
                    ex_ys = generator([ex_rows], which_block=args.layer, post_model=True).detach()
                    ex_ys = torch.clamp(_tanh_to_sigmoid(ex_ys), min=0.0, max=1.0)
                    ex_images.append(ex_ys)
                ex_images = torch.cat(ex_images, dim=0)
                torchvision.utils.save_image(ex_images.detach().cpu(), os.path.join(out_dir,
                                                                                    'layer_%d_class_%d_iter_%d.png' %
                                                                                    (args.layer, class_i, iter)),
                                             nrow=image_number)
                # the subspace mask of class_i, save it.
                subspace_i = predict == class_i
                with open(os.path.join(out_dir, 'subspace_mask_layer%d_iter%d_class%d.pkl' %
                                                (args.layer, iter, class_i)), 'wb') as file_out:
                    pkl.dump(subspace_i, file_out)
                    print('Save layer %d iter %d class %d, out_dir=%s.' % (args.layer, iter, class_i, out_dir))

        if iter % args.report_model == 0:
            with open(os.path.join(out_dir, 'value%d_layer%d.pkl'%(iter, args.layer)), 'wb') as file_out:
                pkl.dump(Matrix.detach().cpu().numpy(), file_out)
                print('Save S.')

            with open(os.path.join(out_dir, 'loss_dict%d_layer%d.pkl'%(iter, args.layer)), 'wb') as file_out:
                pkl.dump(losses_dict, file_out)
                print('Save dict.')


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
    parser.add_argument('-o', '--outputs',
                        default='./TRAIN2',
                        help='Path to save results.')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Layerwise',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # Generator Setting, Check models/model_settings for available GAN models
    parser.add_argument('--gan_model', default='pggan_celebahq',
                        help='The name of model used.', type=str)
    # Loss Parameters
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Size of images for perceptual model')

    parser.add_argument('--optimization', default='Adam',
                        help="['GD', 'Adam']. Optimization method used.")
    parser.add_argument('--lr', default=1e-4,
                        help='Learning rate.', type=float)
    parser.add_argument('--sparse_type', default='L2', type=str, help='the type of sparse term.')
    parser.add_argument('--iterations', default=10000,
                        help='Number of optimization steps.', type=int)
    parser.add_argument('--batch_size', default=4,
                        type=int, help='The number of batchsize')
    parser.add_argument('--layer', default=3,
                        type=int, help='The number of batchsize')
    parser.add_argument('--exp_id', default="SubspaceGAN2",
                        type=str, help='The prefix of the experiment. ')
    parser.add_argument('--beta', default=1.0, type=float, help='...')

    parser.add_argument('--n_subs', type=int, default=6, help='The number of cluster')
    parser.add_argument('--d_subs', type=int, default=6, help='The number of subspace dimension.')
    parser.add_argument('--power', type=float, default=3.0, help='The power of the alpha.')
    parser.add_argument('--alpha', type=float, default=0.2, help='The power of the alpha.')
    parser.add_argument('--times', type=float, default=3, help='the times')

    # Video Settings
    parser.add_argument('--video', type=str2bool, default=False, help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate of the created video.')

    parser.add_argument('--report_image', type=int, default=50)
    parser.add_argument('--report_value', type=int, default=10)
    parser.add_argument('--report_model', type=int, default=100)
    parser.add_argument('--beta0', default=1.0,
                        help='Learning rate.', type=float)
    parser.add_argument('--beta1', default=1.0,
                        help='Learning rate.', type=float)
    args = parser.parse_args()

    main(args)

    # CUDA_VISIBLE_DEVICES=0 python self_express_gan.py --target_images=./examples/gan_inversion/bedroom/
    # --outputs=./TRAIN --inversion_type=PGGAN-Layerwise --gan_model pggan_tower --layer=3 --iterations=20000
    # --optimization=Adam --lr=0.0001 --report_image=100 --report_model=1000 --batch_size=4 --exp_id=Express
    # --init_type=Zeros --beta=0.1 --K=6 --alpha=0.2 --d=3

    # CUDA_VISIBLE_DEVICES=0 python self_express_gan.py --target_images=./examples/face/ --outputs=./TRAIN --inversion_type=PGGAN-Layerwise --gan_model=pggan_celebahq --layer=3 --iterations=20000 --optimization=Adam --lr=0.0001 --report_image=5 --report_model=5 --batch_size=4 --exp_id=SelfExpress --init_type=Zeros --beta=1 --cluster_numbers=6 --alpha=0.2 --d=6 --sparse_type=L1 --power=2.0
    # CUDA_VISIBLE_DEVICES=0 python self_express_gan.py --target_images=./examples/face/ --outputs=./TRAIN --inversion_type=PGGAN-Layerwise --gan_model=pggan_celebahq --layer=3 --iterations=20000 --optimization=Adam --lr=0.0001 --report_image=5 --report_model=5 --batch_size=4 --exp_id=SelfExpress --init_type=Zeros --beta=1 --cluster_numbers=6 --alpha=0.2 --subspace_dimension=6 --sparse_type=L1 --power=2.0
