"""
Use high-level API of timm, create_model(), which is more stable and allows using always the latest version
of timm and the newest pretrained models.

The EfficientNetBaseEncoder from timm_efficientnet.py has a "hidden_layer" = head w/o output (FC) layer,
but its forward() method is not calling it (only extracts stage features).
timm's classes don't provide that, only (depending on `features_only`)
    (1) the normal EfficientNet class, which calls head layers explicitly and returns no stage features
    (2) EfficientNetFeatures class, which extracts all stage features but completely lacks the head.
Hence, when using (2), the "hidden_layer" must be constructed from scratch in PretrainedModel.

ToDo: can encoder with features_only=True replace the entire PretrainModel class?
"""

from functools import partial

import timm

timm_default_encoders = {name: {'encoder': partial(timm.create_model, model_name=name),
                                'pretrained_settings': {},
                                'params': {'features_only': True},
                                } for name in timm.list_models()}