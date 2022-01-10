# python 3.7
"""Helper function to build generator."""

from .model_settings import MODEL_POOL
from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan2_generator import StyleGAN2Generator
from .biggan_generator import BigGANDeepGenerator
from .biggan_generator import BigGANShallowGenerator
from .sngan_generator import SNGANGenerator
from .stylegan2_discriminator import StyleGAN2Discriminator

__all__ = ['build_generator']


def build_generator(model_name, logger=None):
  """ Builds generator module by model name."""
  if not model_name in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = MODEL_POOL[model_name]['gan_type']

  if gan_type == 'pggan':
    return PGGANGenerator(model_name, logger=logger)
  if gan_type == 'pggan_disc':
    return PGGANDiscriminator(model_name, logger=logger)
  if gan_type == 'stylegan':
    return StyleGANGenerator(model_name, logger=logger)
  if gan_type == 'stylegan2':
    return StyleGAN2Generator(model_name, logger=logger)
  if gan_type == 'stylegan2_disc':
    return StyleGAN2Discriminator(model_name, logger=logger)
  if gan_type == 'biggandeep':
    return BigGANDeepGenerator(model_name, logger=logger)
  if gan_type == 'bigganshallow':
    return BigGANShallowGenerator(model_name, logger=logger)
  if gan_type == 'sngan':
    return SNGANGenerator(model_name, logger=logger)

  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')
