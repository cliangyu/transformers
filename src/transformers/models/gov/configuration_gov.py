# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GOV model configuration"""

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING
from ..gpt_oss import GptOssConfig
from ..internvl import InternVLVisionConfig


class GovVisionConfig(InternVLVisionConfig):
    """
    This is the configuration class to store the configuration of a GOV vision model.
    It inherits from InternVLVisionConfig to reuse the vision encoder architecture.
    """

    model_type = "gov_vision"
    base_config_key = "vision_config"


class GovConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GovForConditionalGeneration`].
    It is used to instantiate a GOV model according to the specified arguments, defining the model architecture.
    GOV combines GPT-OSS as the language model with InternVL's vision encoder.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[GovVisionConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone.
        text_config (`Union[GptOssConfig, dict]`, *optional*):
            The config object or dictionary of the GPT-OSS text backbone.
        image_token_id (`int`, *optional*, defaults to 151667):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 256):
            Number of image tokens to use per image patch.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            Factor by which to downsample the image features.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the projector.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to use as the image features.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.

    Example:

    ```python
    >>> from transformers import GovConfig, GovForConditionalGeneration

    >>> # Initializing a GOV style configuration
    >>> configuration = GovConfig()

    >>> # Initializing a model from the configuration
    >>> model = GovForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gov"
    sub_configs = {"text_config": GptOssConfig, "vision_config": GovVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151667,
        image_seq_length=256,
        downsample_ratio=0.5,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        # Initialize vision config
        if isinstance(vision_config, dict):
            self.vision_config = GovVisionConfig(**vision_config)
        elif isinstance(vision_config, GovVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = GovVisionConfig()
        else:
            raise ValueError(f"vision_config must be a dict, GovVisionConfig, or None, got {type(vision_config)}")

        # Initialize text config (GPT-OSS)
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "gpt_oss")
            self.text_config = GptOssConfig(**text_config)
        elif isinstance(text_config, GptOssConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = GptOssConfig()
        else:
            raise ValueError(f"text_config must be a dict, GptOssConfig, or None, got {type(text_config)}")

        super().__init__(**kwargs)
