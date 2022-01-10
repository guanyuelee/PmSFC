# python 3.7
"""Contains the generator class of StyleGAN2.

This class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator, BaseDiscriminator
from .stylegan2_discriminator_network import StyleGAN2DiscriminatorNet, discriminator_fill_statedict

__all__ = ['StyleGAN2Discriminator']


class StyleGAN2Discriminator(BaseDiscriminator):
  """Defines the generator class of StyleGAN2.

  Same as StyleGAN, StyleGAN2 also has Z space, W space, and W+ (WP) space.
  """

  def __init__(self, model_name, logger=None):
    super().__init__(model_name, logger)
    assert self.gan_type == 'stylegan2_disc'

  def build(self):
    self.check_attr('d_architecture_type')
    self.check_attr('fused_modulate')
    self.net = StyleGAN2DiscriminatorNet(
        size=self.resolution,
        image_channels=self.image_channels,
        architecture_type=self.d_architecture_type)
    self.num_layers = self.net.num_layers
    self.model_specific_vars = []

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

    for key, val in tf_vars.items():
      print(key, val.shape)

    state_dict = self.net.state_dict()
    state_dict = discriminator_fill_statedict(state_dict, D.vars, size=self.resolution)
    self.logger.info(f'Successfully converted!')

    for key, val in state_dict.items():
      print(key, val.shape)

    self.logger.info(f'Saving pytorch weights to `{self.weight_path}`.')
    for var_name in self.model_specific_vars:
      del state_dict[var_name]
    torch.save(state_dict, self.weight_path)
    self.logger.info(f'Successfully saved!')

    self.load()

    # Start testing if needed.
    if test_num <= 0 or not tf.test.is_built_with_cuda():
      self.logger.warning(f'Skip testing the weights converted from tf model!')
      sess.close()
      return
    self.logger.info(f'Testing conversion results.')
    self.net.eval().to(self.run_device)
    total_distance = 0.0
    for i in range(test_num):
      latent_code = self.easy_sample(1)
      tf_output = D.run(latent_code, None)
      print(tf_output)
      pth_output = self.net(torch.from_numpy(latent_code).cuda())[-1].detach().cpu().numpy()
      print(pth_output)
      distance = np.average(np.abs(tf_output - pth_output))
      self.logger.debug(f'  Test {i:03d}: distance {distance:.6e}.')
      total_distance += distance
    self.logger.info(f'Average distance is {total_distance / test_num:.6e}.')

    sess.close()

  def sample(self, num, latent_space_type='z', **kwargs):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.
      latent_space_type: Type of latent space from which to sample latent code.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)

    Returns:
      A `numpy.ndarray` as sampled latend codes.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    res = np.random.random((num, self.image_channels, self.resolution, self.resolution))

    return res.astype(np.float32)

  def preprocess(self, latent_codes, latent_space_type='z', **kwargs):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    return latent_codes

  def _synthesize(self,
                  latent_codes,
                  latent_space_type='z',
                  generate_style=False,
                  generate_image=True):
    """Synthesizes images with given latent codes.

    One can choose whether to generate the layer-wise style codes.

    Args:
      latent_codes: Input latent codes for image synthesis.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`z`, `w`, `wp`] are supported. Case insensitive. (default: `z`)
      generate_style: Whether to generate the layer-wise style codes. (default:
        False)
      generate_image: Whether to generate the final image synthesis. (default:
        True)

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}

    latent_space_type = latent_space_type.lower()
    # Generate from Z space.
    if latent_space_type == 'z':
      if not (len(latent_codes.shape) == 2 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.z_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'latent_space_dim], where `batch_size` no larger '
                         f'than {self.batch_size}, and `latent_space_dim` '
                         f'equal to {self.z_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      zs = zs.to(self.run_device)
      ws = self.net.mapping(zs)
      wps = self.net.truncation(ws)
      results['z'] = latent_codes
      results['w'] = self.get_value(ws)
      results['wp'] = self.get_value(wps)
    # Generate from W space.
    elif latent_space_type == 'w':
      if not (len(latent_codes.shape) == 2 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.w_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'w_space_dim], where `batch_size` no larger than '
                         f'{self.batch_size}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      ws = ws.to(self.run_device)
      wps = self.net.truncation(ws)
      results['w'] = latent_codes
      results['wp'] = self.get_value(wps)
    # Generate from W+ space.
    elif latent_space_type == 'wp':
      if not (len(latent_codes.shape) == 3 and
              0 < latent_codes.shape[0] <= self.batch_size and
              latent_codes.shape[1] == self.num_layers and
              latent_codes.shape[2] == self.w_space_dim):
        raise ValueError(f'Latent codes should be with shape [batch_size, '
                         f'num_layers, w_space_dim], where `batch_size` no '
                         f'larger than {self.batch_size}, `num_layers` equal '
                         f'to {self.num_layers}, and `w_space_dim` equal to '
                         f'{self.w_space_dim}!\n'
                         f'But {latent_codes.shape} received!')
      wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      wps = wps.to(self.run_device)
      results['wp'] = latent_codes
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    if generate_style:
      for i in range(self.num_layers - 1):
        style = self.net.synthesis.__getattr__(f'layer{i}').style(wps[:, i, :])
        results[f'style{i:02d}'] = self.get_value(style)
      style = self.net.synthesis.__getattr__(
          f'output{i // 2}').style(wps[:, i + 1, :])
      results[f'style{i + 1:02d}'] = self.get_value(style)

    if generate_image:
      images = self.net.synthesis(wps)
      results['image'] = self.get_value(images)

    if self.use_cuda:
      torch.cuda.empty_cache()

    return results

  def synthesize(self,
                 latent_codes,
                 latent_space_type='z',
                 generate_style=False,
                 generate_image=True):
    return self.batch_run(latent_codes,
                          lambda x: self._synthesize(
                              x,
                              latent_space_type=latent_space_type,
                              generate_style=generate_style,
                              generate_image=generate_image))
