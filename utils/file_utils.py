import os
from PIL import Image
import torchvision
import torch
import pickle as pkl
import numpy as np

from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize


IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm']
INCLUDED_PARAMETERS = ['generator', 'optimization', 'inversion_type', 'gan_model', 'lr', 'p_lr',
                       'init_type', 'beta', 'code_per_cluster', 'layer', 'composing_layer', 'beta0',
                       'hidden_channels', 'normal_feature']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
    assert _is_image_file(path), "%s is not an image" % path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def load_as_tensor(path, mode='RGB'):
    """
    Load image to tensor
    :param path: image path
    :param mode: 'Y' returns 1 channel tensor, 'RGB' returns 3 channels, 'RGBA' returns 4 channels, 'YCbCr' returns 3 channels
    :return: 3D tensor
    """
    if mode != 'Y':
        return PIL2Tensor(pil_loader(path, mode=mode))
    else:
        return PIL2Tensor(pil_loader(path, mode='YCbCr'))[:1]


def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return torchvision.transforms.functional.to_pil_image(tensor_image.detach(), mode=mode)


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def image_files(path):
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return image_files


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def easy_prepare_directory(args):
    if not os.path.exists(args.outputs):
        os.mkdir(args.outputs)
        print(f'Create directory: {args.outputs} successfully.')
    if "_" in args.gan_model:
        gan_type, dataset = args.gan_model.split("_")
    else:
        gan_type = args.gan_model
        dataset = args.gan_model

    directory = os.path.join(args.outputs, gan_type)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f'Create directory: {directory} successfully.')

    directory = os.path.join(directory, dataset)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f'Create directory: {directory} successfully.')

    return directory


def create_experiments_directory(args, pref):
    directory = easy_prepare_directory(args)

    args_dict = []
    for x, y in sorted(vars(args).items()):
        if x in INCLUDED_PARAMETERS:
            args_dict.append(x + str(y))

    # args_dict = [x + str(y) if x in INCLUDED_PARAMETERS else "" for x, y in sorted(vars(args).items())]
    print(args_dict)
    experiment_name = "_".join([pref, ] + args_dict)

    if not os.path.exists(os.path.join(directory, experiment_name)):
        os.mkdir(os.path.join(directory, experiment_name))
        print('Create experiment\'%s\'' % experiment_name)

    return os.path.join(directory, experiment_name), experiment_name


TRANSFORMER_PARAMETERS = ['layer', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio',
                          'drop_rate', 'attn_drop_rate', 'm_scale', 'wgt_reg',
                          'truncated_bound', 'n_dirs', 'truncated', 'wgt_dens',
                          'n_layers', 'hidden_dim', 't_scale', 'wgt_orth', 'wgt_pos', 'wgt_neg',
                          'which_layers', 'beta_d', 'beta_f', 'beta_sparse', 'cluster_numbers',
                          'which_dir']


def create_transformer_experiments_directory(args, pref):
    directory = easy_prepare_directory(args)

    args_dict = []
    for x, y in sorted(vars(args).items()):
        if x in TRANSFORMER_PARAMETERS:
            args_dict.append(x + str(y))

    # args_dict = [x + str(y) if x in INCLUDED_PARAMETERS else "" for x, y in sorted(vars(args).items())]
    print(args_dict)
    experiment_name = "_".join([pref, ] + args_dict)

    if not os.path.exists(os.path.join(directory, experiment_name)):
        os.mkdir(os.path.join(directory, experiment_name))
        print('Create experiment\'%s\'' % experiment_name)

    return os.path.join(directory, experiment_name), experiment_name


def easy_create_directory(directory, file):
    directory = os.path.join(directory, file)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f'Create directory: {directory} successfully.')
    return directory


def create_feature_experiments_directory(args, prefix, included_lists):
    # create the experiment directory for feature extraction experiments.
    if not os.path.exists(args.outputs):
        os.mkdir(args.outputs)
        print(f'Create directory: {args.outputs} successfully.')

    if "_" in args.gan_model:
        gan_type, dataset = args.gan_model.split("_")
    else:
        gan_type = args.gan_model
        dataset = args.gan_model

    directory = easy_create_directory(args.outputs, gan_type)
    directory = easy_create_directory(directory, dataset)

    # create experiment name
    args_dict = []
    for x, y in sorted(vars(args).items()):
        if x in included_lists:
            args_dict.append(x + str(y))
    print(args_dict)

    experiment_name = "_".join([prefix, ] + args_dict)
    path_to_exp = os.path.join(directory, experiment_name)
    os.makedirs(path_to_exp, exist_ok=True)

    return os.path.join(directory, experiment_name), experiment_name


def check_transformer_experiments_directory(args, pref):
    directory = easy_prepare_directory(args)

    args_dict = []
    for x, y in sorted(vars(args).items()):
        if x in TRANSFORMER_PARAMETERS:
            args_dict.append(x + str(y))

    # args_dict = [x + str(y) if x in INCLUDED_PARAMETERS else "" for x, y in sorted(vars(args).items())]
    print(args_dict)
    experiment_name = "_".join([pref, ] + args_dict)

    if not os.path.exists(os.path.join(directory, experiment_name)):
        # os.mkdir(os.path.join(directory, experiment_name))
        raise NotADirectoryError('There isn\'t a directory \'%s\', please check the arguments provided. ' % (experiment_name))

    return os.path.join(directory, experiment_name), experiment_name


SUBSPACE_PARAMETERS = ['n_subspaces', 's_batch_size', 'wgt_f', 'wgt_x', 's_layer', 'wgt_spa']


def create_subspace_distill_directory(which_path, args, exp_id):
    args_dict = []
    for x, y in sorted(vars(args).items()):
        if x in SUBSPACE_PARAMETERS:
            args_dict.append(x + str(y))

    print(args_dict)
    second_experiment_name = "_".join([exp_id, ] + args_dict)

    path = os.path.join(which_path, second_experiment_name)
    if not os.path.exists(path):
        os.mkdir(path)
        print('create directory: %s.' % path)

    return path, second_experiment_name


# return the latest checkpoints if not specified by restore_step.
def restore_saved_checkpoints(save_dir, restore_step=-1, eps=1e-16):
    file_list = os.listdir(save_dir)
    print('There are %d saved checkpoints in %s.' % (len(file_list), save_dir))
    if restore_step == -1:
        print('Note: You do not specify which checkpoint to restore, so we restore the latest one by default. ')
        step_lists = []
        for i in range(len(file_list)):
            item = file_list[i].split('_')[3]
            step = int(item.split('.')[0])
            step_lists.append(step)
        max_idx = np.argmax(step_lists)
        restore_step = step_lists[max_idx]
        print('Note: We will set restore step = %d.' % restore_step)

    restore_file = 'fmap_transformer_iter_%d.pth' % restore_step
    save_dicts = torch.load(os.path.join(save_dir, restore_file))
    D = save_dicts['D']

    # check if D is normalized.
    norm_D = torch.sqrt(torch.sum(D * D, dim=1) + eps) - 1.0
    for i in range(norm_D.shape[0]):
        assert norm_D[i] < 1e-3

    return D


def load_DPT(traverser, dim_size, m_scale):
    DPT_DIRECTORY = './bin/trained-traverser/'
    path = os.path.join(DPT_DIRECTORY, 'trained-traverser-mscale-%d-dim-%d.pth' % (m_scale, dim_size))
    state_dict = torch.load(path)
    traverser.load_state_dict(state_dict['directions'])
    print('Note: load traverser from %s successfully. ' % path)


def prepare_test_z(args, n_test_zs=50):
    gan_type, image_type = args.gan_model.split("_")
    if args.gan_model.startswith('pggan'):
        test_image_dir = os.path.join('./bin/', gan_type, image_type)
        files = os.listdir(test_image_dir)
        test_zs = []
        for i in range(len(files)):
            if files[i].endswith('.pkl'):
                with open(os.path.join(test_image_dir, files[i]), 'rb') as file_in:
                    test_zs.append(pkl.load(file_in))
        test_zs = torch.from_numpy(np.concatenate(test_zs, axis=0).astype(np.float32)).cuda()
    elif args.gan_model.startswith('stylegan'):
        test_image_dir = os.path.join('./bin/test_zs/', 'all_zs_%s.pkl' % args.gan_model)
        with open(test_image_dir, 'rb') as file_in:
            test_zs = pkl.load(file_in)
            print('Load test_z with shape, ', test_zs.shape)
            test_zs = torch.from_numpy(test_zs).cuda()
        return test_zs
    elif args.gan_model.startswith('biggan'):
        test_image_dir = os.path.join('./bin/test_zs/', 'all_zs_biggandeep256_imagenet.pkl')
        with open(test_image_dir, 'rb') as file_in:
            test_zs = pkl.load(file_in)
            print('Load test_z with shape, ', test_zs.shape)
            test_zs = torch.from_numpy(test_zs).cuda()
        return test_zs
    else:
        # raise NotImplemented()
        test_zs = torch.randn(size=(n_test_zs, args.dim_z, 1, 1), dtype=torch.float32).cuda()
    return test_zs


def get_generator_info(args, generator, which_layer=-1):
    # build two transformers
    if which_layer == -1:
        which_layer = args.layer

    if args.gan_model.startswith('pggan'):
        latent_space = generator.module.PGGAN_LATENT[which_layer]
        output_size = generator.module.PGGAN_LATENT[-1]
        fmap_size = latent_space[1]
        fmap_ch = latent_space[0]
        image_size = output_size[1]
        image_ch = 3
    elif args.gan_model.startswith('biggandeep'):
        z_debug = torch.randn(size=(args.n_samples, args.dim_z, 1, 1), dtype=torch.float32)
        y_debug = torch.from_numpy(np.ones(shape=(args.n_samples, 1), dtype=np.int64) * args.which_class)
        print(z_debug.shape)
        print(y_debug.shape)
        f_debug = generator([z_debug, y_debug], truncation=args.truncation,
                            which_block=which_layer, pre_model=True).detach()
        print(f_debug.shape)
        if len(f_debug.shape) == 4:
            latent_space = f_debug.shape[1:]
            x_debug = generator([z_debug, y_debug], features=f_debug, truncation=args.truncation,
                                which_block=which_layer, post_model=True).detach()
            output_size = x_debug.shape[1:]
            print(output_size)
            fmap_size = latent_space[1]
            fmap_ch = latent_space[0]
            image_size = output_size[1]
            image_ch = 3
        elif len(f_debug.shape) == 2:
            latent_space = f_debug.shape[1:]
            x_debug = generator([z_debug, y_debug], features=f_debug, truncation=args.truncation,
                                which_block=which_layer, post_model=True).detach()
            output_size = x_debug.shape[1:]
            print(output_size)
            fmap_size = 1
            fmap_ch = latent_space[0]
            image_size = output_size[1]
            image_ch = 3

    else:
        z_debug = torch.randn(size=(args.n_samples, args.dim_z, 1, 1), dtype=torch.float32)
        f_debug = generator([z_debug], which_block=which_layer, pre_model=True).detach()
        if len(f_debug.shape) == 4:
            latent_space = f_debug.shape[1:]
            x_debug = generator([f_debug], which_block=which_layer, post_model=True).detach()
            output_size = x_debug.shape[1:]
            fmap_size = latent_space[1]
            fmap_ch = latent_space[0]
            image_size = output_size[1]
            image_ch = 1 if 'mnist' in args.gan_model else 3
        elif len(f_debug.shape) == 2:
            latent_space = f_debug.shape[1:]
            x_debug = generator([f_debug], which_block=which_layer, post_model=True).detach()
            output_size = x_debug.shape[1:]
            fmap_size = 1
            fmap_ch = latent_space[0]
            image_size = output_size[1]
            image_ch = 1 if 'mnist' in args.gan_model else 3
        elif len(f_debug.shape) == 3:
            latent_space = f_debug.shape[1:]
            x_debug = generator([f_debug], which_block=which_layer, post_model=True).detach()
            output_size = x_debug.shape[1:]
            fmap_size = 1
            fmap_ch = (latent_space[0], latent_space[1])
            image_size = output_size[1]
            image_ch = 1 if 'mnist' in args.gan_model else 3
            return fmap_size, fmap_ch, image_size, image_ch

    return fmap_size, fmap_ch, image_size, image_ch


def get_sorted_subspace_prediction(predict, args):
    p_sum = [sum(predict == k) for k in range(1, args.n_subspaces + 1, 1)]
    p_sum = np.array(p_sum)
    p_sort = np.argsort(p_sum)[::-1]
    predict_new = predict.copy()
    for i in range(1, args.n_subspaces + 1, 1):
        predict_new[predict == (p_sort[i - 1] + 1)] = i
    p_sum = [sum(predict_new == k) for k in range(1, args.n_subspaces + 1, 1)]
    print(p_sum)
    return predict_new.copy(), p_sum


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


def get_dir_lists(dir_lists):
    split1 = dir_lists.split('+')
    results = []
    for item in split1:
        if ':' in item:
            split2 = item.split(':')
            start =int(split2[0])
            end = int(split2[1])
            split2 = [x for x in range(start, end, 1)]
            results += split2
        else:
            split3 = [int(x) for x in item.split('_')]
            results += split3
    return results


def get_images_from_path(path):
    supported_image_ext = ['png', 'jpg', 'jpeg']
    lists = os.listdir(path)
    lists_filtered = []

    for item in lists:
        try:
            ext = item.split('.')[1]
            if ext in supported_image_ext:
                lists_filtered.append(os.path.join(path, item))
        except:
            print(f'{item} is not a valid image. ')

    return np.array(lists_filtered)


def read_images_from_pathes(lists, return_tensor=False, return_cuda=False):
    images = []
    for file in lists:
        img = np.array(Image.open(file)).astype(np.float32)
        img = img / 255.0
        img = np.transpose(img, axes=[2, 0, 1])
        s = img.shape
        img = np.reshape(img, newshape=[1, s[0], s[1], s[2]])
        images.append(img)

    images = np.concatenate(images, axis=0)

    if return_tensor:
        return torch.from_numpy(images)

    if return_cuda:
        return torch.from_numpy(images).cuda()


def get_parameter_count(params_list):
    total = 0
    for module in params_list:
        part = sum([param.nelement() for param in module.parameters()])
        total += part
    return total


def calculate_psnr(image1, image2):
    max_val = 255
    y = torch.mean(torch.pow(image1 * 255 - image2 * 255, 2.0), dim=[1, 2, 3])
    y = 20 * torch.log10(max_val/(torch.sqrt(y)))
    return torch.mean(y).cpu().detach().numpy()


def calculate_lpips(image1, image2, model):
    res = []
    for i in range(image1.shape[0]):
        img1 = image1[i:i+1]
        img2 = image2[i:i+1]
        y = model.forward(img1, img2).cpu().detach().numpy()
        res.append(y)
    return np.mean(res)



















