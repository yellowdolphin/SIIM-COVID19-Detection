from functools import partial

import torch
import torch.nn as nn

import timm

#from timm.models.efficientnet import EfficientNet
#from timm.models.efficientnet import decode_arch_def, round_channels, default_cfgs
#from timm.models.layers.activations import Swish

#from ._base import EncoderMixin


def prepare_settings(settings):
    return {
        "mean": settings["mean"],
        "std": settings["std"],
        "url": settings["url"],
        "input_range": (0, 1),
        "input_space": "RGB",
    }

timm_default_encoders = {name: {'encoder': partial(timm.create_model, model_name=name),
                                'pretrained_settings': {},
                                'params': {'features_only': True}} for name in timm.list_models()}
