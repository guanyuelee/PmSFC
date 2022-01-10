# python 3.7
"""Contains the generator class of PGGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np
from itertools import chain

import torch
import logging

from .base_generator import BaseGenerator
from .biggan_generator_network import BigGANDeepGeneratorNet, BigGANConfig
from torch.nn.functional import normalize
from .gan_load import make_unconditioned_big_gan
from models.BigGAN import BigGAN

__all__ = ['BigGANDeepGenerator', 'BigGANShallowGenerator']

logger = logging.getLogger(__name__)


class BigGANDeepGenerator(BaseGenerator):
    """Defines the generator class of PGGAN."""

    def __init__(self, model_name, logger=None):
        super().__init__(model_name, logger)
        assert self.gan_type == 'biggandeep'
        self.model_name = model_name
        # self.lod = self.net.lod.to(self.cpu_device).tolist()
        # self.logger.info(f'Current `lod` is {self.lod}.')

    def build(self):
        logger.info("loading model {} from cache at {}".format(self.model_name, self.weight_path))

        # Load config
        config = BigGANConfig.from_json_file(self.config_path)
        self.config = config
        logger.info("Model config {}".format(config))

        self.net = BigGANDeepGeneratorNet(config)
        # state_dict = torch.load(self.weight_path,
        #                         map_location='cpu' if not torch.cuda.is_available() else None)
        # self.net.load_state_dict(state_dict, strict=False)

    def sample(self, num, **kwargs):
        assert num > 0
        return np.random.randn(num, self.z_space_dim).astype(np.float32)

    def preprocess(self, latent_codes, **kwargs):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        latent_codes = latent_codes.reshape(-1, self.z_space_dim)
        norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
        latent_codes = latent_codes / norm * np.sqrt(self.z_space_dim)
        return latent_codes.astype(np.float32)

    def _synthesize(self, latent_codes):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
        if not (len(latent_codes.shape) == 2 and
                0 < latent_codes.shape[0] <= self.batch_size and
                latent_codes.shape[1] == self.z_space_dim):
            raise ValueError(f'Latent codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger than '
                             f'{self.batch_size}, and `latent_space_dim` equals to '
                             f'{self.z_space_dim}!\n'
                             f'But {latent_codes.shape} received!')

        zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
        zs = zs.to(self.run_device)
        images = self.net(zs)
        results = {
            'z': latent_codes,
            'image': self.get_value(images),
        }

        if self.use_cuda:
            torch.cuda.empty_cache()

        return results

    def synthesize(self, latent_codes, **kwargs):
        return self.batch_run(latent_codes, self._synthesize)

    def convert_tf_weights(self, test_num=10):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            raise ImportError("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
                              "https://www.tensorflow.org/install/ for installation instructions.")
        # Load weights from TF model
        checkpoint_path = self.tf_weight_path + "/variables/variables"
        init_vars = tf.train.list_variables(checkpoint_path)
        from pprint import pprint
        pprint(init_vars)

        # Extract batch norm statistics from model if needed
        if self.batch_norm_stats_path:
            stats = torch.load(self.batch_norm_stats_path)
        else:
            logger.info("Extracting batch norm stats")
            stats = extract_batch_norm_stats(self.batch_norm_stats_path)

        # Build TF to PyTorch weights loading map
        tf_to_pt_map = build_tf_to_pytorch_map(self.net, self.config)

        tf_weights = {}
        for name in tf_to_pt_map.keys():
            array = tf.train.load_variable(checkpoint_path, name)
            tf_weights[name] = array
            # logger.info("Loading TF weight {} with shape {}".format(name, array.shape))

        state_dict = self.net.state_dict()
        # Load parameters
        with torch.no_grad():
            pt_params_pnt = set()
            for name, pointer in tf_to_pt_map.items():
                array = tf_weights[name]
                if pointer.dim() == 1:
                    if pointer.dim() < array.ndim:
                        array = np.squeeze(array)
                elif pointer.dim() == 2:  # Weights
                    array = np.transpose(array)
                elif pointer.dim() == 4:  # Convolutions
                    array = np.transpose(array, (3, 2, 0, 1))
                else:
                    raise "Wrong dimensions to adjust: " + str((pointer.shape, array.shape))
                if pointer.shape != array.shape:
                    raise ValueError("Wrong dimensions: " + str((pointer.shape, array.shape)))
                logger.info("Initialize PyTorch weight {} with shape {}".format(name, pointer.shape))
                #pointer.data = torch.from_numpy(array) if isinstance(array, np.ndarray) else torch.tensor(array)
                state_dict[name] = torch.from_numpy(array) if isinstance(array, np.ndarray) else torch.tensor(array)
                tf_weights.pop(name, None)
                pt_params_pnt.add(pointer.data_ptr())
                # Prepare SpectralNorm buffers by running one step of Spectral Norm (no need to train the model):

            for module in self.model.modules():
                for n, buffer in module.named_buffers():
                    if n == 'weight_v':
                        weight_mat = module.weight_orig
                        weight_mat = weight_mat.reshape(weight_mat.size(0), -1)
                        u = module.weight_u

                        v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.config.eps)
                        buffer.data = v
                        pt_params_pnt.add(buffer.data_ptr())

                        u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.config.eps)
                        module.weight_u.data = u
                        pt_params_pnt.add(module.weight_u.data_ptr())

            # Load batch norm statistics
            index = 0
            for layer in self.model.generator.layers:
                if not hasattr(layer, 'bn_0'):
                    continue
                for i in range(4):  # Batchnorms
                    bn_pointer = getattr(layer, 'bn_%d' % i)
                    pointer = bn_pointer.running_means
                    if pointer.shape != stats[index].shape:
                        raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                    pointer.data = torch.from_numpy(stats[index])
                    pt_params_pnt.add(pointer.data_ptr())

                    pointer = bn_pointer.running_vars
                    if pointer.shape != stats[index + 1].shape:
                        raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                    pointer.data = torch.from_numpy(stats[index + 1])
                    pt_params_pnt.add(pointer.data_ptr())

                    index += 2

            bn_pointer = self.model.generator.bn
            pointer = bn_pointer.running_means
            if pointer.shape != stats[index].shape:
                raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
            pointer.data = torch.from_numpy(stats[index])
            pt_params_pnt.add(pointer.data_ptr())

            pointer = bn_pointer.running_vars
            if pointer.shape != stats[index + 1].shape:
                raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
            pointer.data = torch.from_numpy(stats[index + 1])
            pt_params_pnt.add(pointer.data_ptr())

        remaining_params = list(n for n, t in chain(self.model.named_parameters(), self.model.named_buffers()) \
                                if t.data_ptr() not in pt_params_pnt)

        logger.info("TF Weights not copied to PyTorch model: {} -".format(', '.join(tf_weights.keys())))
        logger.info("Remanining parameters/buffers from PyTorch model: {} -".format(', '.join(remaining_params)))


def extract_batch_norm_stats(tf_model_path, batch_norm_stats_path=None):
    try:
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError("Loading a TensorFlow models in PyTorch, requires TensorFlow and TF Hub to be installed. "
                          "Please see https://www.tensorflow.org/install/ for installation instructions for TensorFlow. "
                          "And see https://github.com/tensorflow/hub for installing Hub. "
                          "Probably pip install tensorflow tensorflow-hub")
    tf.reset_default_graph()
    logger.info('Loading BigGAN module from: {}'.format(tf_model_path))
    module = hub.Module(tf_model_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    stacks = sum(((i*10 + 1, i*10 + 3, i*10 + 6, i*10 + 8) for i in range(50)), ())
    numpy_stacks = []
    for i in stacks:
        logger.info("Retrieving module_apply_default/stack_{}".format(i))
        try:
            stack_var = tf.get_default_graph().get_tensor_by_name("module_apply_default/stack_%d:0" % i)
        except KeyError:
            break  # We have all the stats
        numpy_stacks.append(sess.run(stack_var))

    if batch_norm_stats_path is not None:
        torch.save(numpy_stacks, batch_norm_stats_path)
    else:
        return numpy_stacks


def build_tf_to_pytorch_map(model, config):
    """ Build a map from TF variables to PyTorch modules. """
    tf_to_pt_map = {}

    # Embeddings and GenZ
    tf_to_pt_map.update({'linear/w/ema_0.9999': model.embeddings.weight,
                         'Generator/GenZ/G_linear/b/ema_0.9999': model.generator.gen_z.bias,
                         'Generator/GenZ/G_linear/w/ema_0.9999': model.generator.gen_z.weight_orig,
                         'Generator/GenZ/G_linear/u0': model.generator.gen_z.weight_u})

    # GBlock blocks
    model_layer_idx = 0
    for i, (up, in_channels, out_channels) in enumerate(config.layers):
        if i == config.attention_layer_position:
            model_layer_idx += 1
        layer_str = "Generator/GBlock_%d/" % i if i > 0 else "Generator/GBlock/"
        layer_pnt = model.generator.layers[model_layer_idx]
        for i in range(4):  #  Batchnorms
            batch_str = layer_str + ("BatchNorm_%d/" % i if i > 0 else "BatchNorm/")
            batch_pnt = getattr(layer_pnt, 'bn_%d' % i)
            for name in ('offset', 'scale'):
                sub_module_str = batch_str + name + "/"
                sub_module_pnt = getattr(batch_pnt, name)
                tf_to_pt_map.update({sub_module_str + "w/ema_0.9999": sub_module_pnt.weight_orig,
                                     sub_module_str + "u0": sub_module_pnt.weight_u})
        for i in range(4):  # Convolutions
            conv_str = layer_str + "conv%d/" % i
            conv_pnt = getattr(layer_pnt, 'conv_%d' % i)
            tf_to_pt_map.update({conv_str + "b/ema_0.9999": conv_pnt.bias,
                                 conv_str + "w/ema_0.9999": conv_pnt.weight_orig,
                                 conv_str + "u0": conv_pnt.weight_u})
        model_layer_idx += 1

    # Attention block
    layer_str = "Generator/attention/"
    layer_pnt = model.generator.layers[config.attention_layer_position]
    tf_to_pt_map.update({layer_str + "gamma/ema_0.9999": layer_pnt.gamma})
    for pt_name, tf_name in zip(['snconv1x1_g', 'snconv1x1_o_conv', 'snconv1x1_phi', 'snconv1x1_theta'],
                                ['g/', 'o_conv/', 'phi/', 'theta/']):
        sub_module_str = layer_str + tf_name
        sub_module_pnt = getattr(layer_pnt, pt_name)
        tf_to_pt_map.update({sub_module_str + "w/ema_0.9999": sub_module_pnt.weight_orig,
                             sub_module_str + "u0": sub_module_pnt.weight_u})

    # final batch norm and conv to rgb
    layer_str = "Generator/BatchNorm/"
    layer_pnt = model.generator.bn
    tf_to_pt_map.update({layer_str + "offset/ema_0.9999": layer_pnt.bias,
                         layer_str + "scale/ema_0.9999": layer_pnt.weight})
    layer_str = "Generator/conv_to_rgb/"
    layer_pnt = model.generator.conv_to_rgb
    tf_to_pt_map.update({layer_str + "b/ema_0.9999": layer_pnt.bias,
                         layer_str + "w/ema_0.9999": layer_pnt.weight_orig,
                         layer_str + "u0": layer_pnt.weight_u})
    return tf_to_pt_map


class BigGANShallowGenerator(BaseGenerator):
    """Defines the generator class of PGGAN."""

    def __init__(self, model_name, logger=None):
        super().__init__(model_name, logger)
        assert self.gan_type == 'bigganshallow'
        self.model_name = model_name
        # self.lod = self.net.lod.to(self.cpu_device).tolist()
        # self.logger.info(f'Current `lod` is {self.lod}.')

    def build(self):
        logger.info("loading model {} from cache at {}".format(self.model_name, self.weight_path))

        # Load config
        config = make_unconditioned_big_gan(self.config_path, self.weight_path)
        self.config = config
        logger.info("Model config {}".format(config))

        self.net = BigGAN.Generator(**config)
        # state_dict = torch.load(self.weight_path,
        #                         map_location='cpu' if not torch.cuda.is_available() else None)
        # self.net.load_state_dict(state_dict, strict=False)

    def sample(self, num, **kwargs):
        assert num > 0
        return np.random.randn(num, self.z_space_dim).astype(np.float32)

    def preprocess(self, latent_codes, **kwargs):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        latent_codes = latent_codes.reshape(-1, self.z_space_dim)
        norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
        latent_codes = latent_codes / norm * np.sqrt(self.z_space_dim)
        return latent_codes.astype(np.float32)

    def _synthesize(self, latent_codes):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
        if not (len(latent_codes.shape) == 2 and
                0 < latent_codes.shape[0] <= self.batch_size and
                latent_codes.shape[1] == self.z_space_dim):
            raise ValueError(f'Latent codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger than '
                             f'{self.batch_size}, and `latent_space_dim` equals to '
                             f'{self.z_space_dim}!\n'
                             f'But {latent_codes.shape} received!')

        zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
        zs = zs.to(self.run_device)
        images = self.net(zs)
        results = {
            'z': latent_codes,
            'image': self.get_value(images),
        }

        if self.use_cuda:
            torch.cuda.empty_cache()

        return results

    def synthesize(self, latent_codes, **kwargs):
        return self.batch_run(latent_codes, self._synthesize)

    def convert_tf_weights(self, test_num=10):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            raise ImportError("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
                              "https://www.tensorflow.org/install/ for installation instructions.")
        # Load weights from TF model
        checkpoint_path = self.tf_weight_path + "/variables/variables"
        init_vars = tf.train.list_variables(checkpoint_path)
        from pprint import pprint
        pprint(init_vars)

        # Extract batch norm statistics from model if needed
        if self.batch_norm_stats_path:
            stats = torch.load(self.batch_norm_stats_path)
        else:
            logger.info("Extracting batch norm stats")
            stats = extract_batch_norm_stats(self.batch_norm_stats_path)

        # Build TF to PyTorch weights loading map
        tf_to_pt_map = build_tf_to_pytorch_map(self.net, self.config)

        tf_weights = {}
        for name in tf_to_pt_map.keys():
            array = tf.train.load_variable(checkpoint_path, name)
            tf_weights[name] = array
            # logger.info("Loading TF weight {} with shape {}".format(name, array.shape))

        state_dict = self.net.state_dict()
        # Load parameters
        with torch.no_grad():
            pt_params_pnt = set()
            for name, pointer in tf_to_pt_map.items():
                array = tf_weights[name]
                if pointer.dim() == 1:
                    if pointer.dim() < array.ndim:
                        array = np.squeeze(array)
                elif pointer.dim() == 2:  # Weights
                    array = np.transpose(array)
                elif pointer.dim() == 4:  # Convolutions
                    array = np.transpose(array, (3, 2, 0, 1))
                else:
                    raise "Wrong dimensions to adjust: " + str((pointer.shape, array.shape))
                if pointer.shape != array.shape:
                    raise ValueError("Wrong dimensions: " + str((pointer.shape, array.shape)))
                logger.info("Initialize PyTorch weight {} with shape {}".format(name, pointer.shape))
                #pointer.data = torch.from_numpy(array) if isinstance(array, np.ndarray) else torch.tensor(array)
                state_dict[name] = torch.from_numpy(array) if isinstance(array, np.ndarray) else torch.tensor(array)
                tf_weights.pop(name, None)
                pt_params_pnt.add(pointer.data_ptr())
                # Prepare SpectralNorm buffers by running one step of Spectral Norm (no need to train the model):

            for module in self.model.modules():
                for n, buffer in module.named_buffers():
                    if n == 'weight_v':
                        weight_mat = module.weight_orig
                        weight_mat = weight_mat.reshape(weight_mat.size(0), -1)
                        u = module.weight_u

                        v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.config.eps)
                        buffer.data = v
                        pt_params_pnt.add(buffer.data_ptr())

                        u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.config.eps)
                        module.weight_u.data = u
                        pt_params_pnt.add(module.weight_u.data_ptr())

            # Load batch norm statistics
            index = 0
            for layer in self.model.generator.layers:
                if not hasattr(layer, 'bn_0'):
                    continue
                for i in range(4):  # Batchnorms
                    bn_pointer = getattr(layer, 'bn_%d' % i)
                    pointer = bn_pointer.running_means
                    if pointer.shape != stats[index].shape:
                        raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                    pointer.data = torch.from_numpy(stats[index])
                    pt_params_pnt.add(pointer.data_ptr())

                    pointer = bn_pointer.running_vars
                    if pointer.shape != stats[index + 1].shape:
                        raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                    pointer.data = torch.from_numpy(stats[index + 1])
                    pt_params_pnt.add(pointer.data_ptr())

                    index += 2

            bn_pointer = self.model.generator.bn
            pointer = bn_pointer.running_means
            if pointer.shape != stats[index].shape:
                raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
            pointer.data = torch.from_numpy(stats[index])
            pt_params_pnt.add(pointer.data_ptr())

            pointer = bn_pointer.running_vars
            if pointer.shape != stats[index + 1].shape:
                raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
            pointer.data = torch.from_numpy(stats[index + 1])
            pt_params_pnt.add(pointer.data_ptr())

        remaining_params = list(n for n, t in chain(self.model.named_parameters(), self.model.named_buffers()) \
                                if t.data_ptr() not in pt_params_pnt)

        logger.info("TF Weights not copied to PyTorch model: {} -".format(', '.join(tf_weights.keys())))
        logger.info("Remanining parameters/buffers from PyTorch model: {} -".format(', '.join(remaining_params)))