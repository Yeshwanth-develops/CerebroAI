"""Models module for ResNet-18 classifier and WGAN-GP generator."""
from .resnet18 import build_resnet18
from .wgan_gp import WGAN_GP, build_generator, build_critic

__all__ = ["build_resnet18", "WGAN_GP", "build_generator", "build_critic"]
