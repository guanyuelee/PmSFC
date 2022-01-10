# python 3.7
"""Contains the generator class of PGGAN.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from .base_generator import BaseDiscriminator
from .pggan_discriminator_network import PGGANDiscriminatorNet

__all__ = ['PGGANDiscriminator']


class PGGANDiscriminator(BaseDiscriminator):
    """Defines the discriminator class of PGGAN."""

    def __init__(self, model_name, logger=None):
        super().__init__(model_name, logger)
        assert self.gan_type == 'pggan_disc'
        self.lod = self.net.lod.to(self.cpu_device).tolist()
        self.logger.info(f'Current `lod` is {self.lod}.')

    def build(self):
        self.check_attr('fused_scale')
        self.net = PGGANDiscriminatorNet(resolution=self.resolution,
                                         image_channels=self.image_channels,
                                         fused_scale=self.fused_scale)
        self.num_layers = self.net.num_layers

    def convert_tf_weights(self, test_num=10):
        # pylint: disable=import-outside-toplevel
        import sys
        import pickle
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # pylint: enable=import-outside-toplevel

        sess = tf.compat.v1.InteractiveSession()

        self.logger.info(f'Loading tf weights from `{self.tf_weight_path}`.')
        self.check_attr('tf_code_path')
        sys.path.insert(0, self.tf_code_path)
        with open(self.tf_weight_path, 'rb') as f:
            _, D, _ = pickle.load(f)  # G, D, Gs
        sys.path.pop(0)
        self.logger.info(f'Successfully loaded!')

        self.logger.info(f'Converting tf weights to pytorch version.')
        tf_vars = dict(D.__getstate__()['variables'])
        state_dict = self.net.state_dict()

        for pth_var_name, tf_var_name in self.net.pth_to_tf_var_mapping.items():
            assert tf_var_name in tf_vars
            assert pth_var_name in state_dict
            self.logger.debug(f'  Converting `{tf_var_name}{tf_vars[tf_var_name].shape}` to `{pth_var_name}{state_dict[pth_var_name].shape}`.')
            var = torch.from_numpy(np.array(tf_vars[tf_var_name]))
            if 'weight' in pth_var_name:
                if 'conv' in pth_var_name:
                    var = var.permute(3, 2, 0, 1)
                elif 'dense' in pth_var_name:
                    var = var.permute(1, 0)

            state_dict[pth_var_name] = var
        self.logger.info(f'Successfully converted!')

        self.logger.info(f'Saving pytorch weights to `{self.weight_path}`.')
        for var_name in self.model_specific_vars:
            del state_dict[var_name]
        torch.save(state_dict, self.weight_path, _use_new_zipfile_serialization=False)
        self.logger.info(f'Successfully saved!')
        self.load()

        # Start testing if needed.
        if test_num <= 0 or not tf.test.is_built_with_cuda():
            self.logger.warning(f'Skip testing the weights converted from tf model!')
            sess.close()
            return
        self.logger.info(f'Testing conversion results.')
        self.net.eval().to(self.run_device)
        # input_shapes = D.input_shapes
        # tf_fake_label = np.random.rand(4, input_shapes[1], input_shapes[2], input_shapes[3]).astype(np.float32)
        total_distance = 0.0
        for i in range(test_num):
            images = self.easy_sample(1)
            tf_output = D.run(images)
            pth_output = self.synthesize(images)['output']
            distance = np.average(np.abs(tf_output - pth_output))
            self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
            total_distance += distance
        self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

        sess.close()

    def sample(self, num, **kwargs):
        assert num > 0
        input_shapes = self.net.input_shapes
        return np.random.rand(num, input_shapes[1], input_shapes[2], input_shapes[3]).astype(np.float32)

    def preprocess(self, latent_codes, **kwargs):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

        result = np.clip(2.0 * latent_codes - 1.0, -1.0, 1.0)
        return result.astype(np.float32)

    def _synthesize(self, images):
        if not isinstance(images, np.ndarray):
            raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')
        if not (len(images.shape) == 4 and
                0 < images.shape[0] <= self.batch_size and
                images.shape[1] == self.image_channels and images.shape[2] == self.resolution):
            raise ValueError(f'Latent codes should be with shape [batch_size, '
                             f'latent_space_dim], where `batch_size` no larger than '
                             f'{self.batch_size}, and `latent_space_dim` equals to '
                             f'{self.resolution}!\n'
                             f'But {images.shape} received!')

        images = torch.from_numpy(images).type(torch.FloatTensor)
        images = images.to(self.run_device)
        outputs = self.net(images)
        results = {
            'images': images.detach().cpu().numpy(),
            'output': outputs.detach().cpu().numpy(),
        }

        if self.use_cuda:
            torch.cuda.empty_cache()

        return results

    def synthesize(self, latent_codes, **kwargs):
        return self.batch_run(latent_codes, self._synthesize)
