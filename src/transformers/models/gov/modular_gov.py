# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""GOV model: GPT-OSS + InternVL Vision"""

import collections.abc
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...utils import logging
from ..internvl.configuration_internvl import InternVLConfig, InternVLVisionConfig
from ..internvl.modeling_internvl import (
    InternVLCausalLMOutputWithPast,
    InternVLForConditionalGeneration,
    InternVLModel,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLPreTrainedModel,
    InternVLVisionModel,
    InternVLVisionPreTrainedModel,
)


logger = logging.get_logger(__name__)


class GovVisionConfig(InternVLVisionConfig):
    pass


class GovConfig(InternVLConfig):
    """
    Configuration class for GOV model that combines GPT-OSS text model with InternVL vision.
    """

    model_type = "gov"

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
        # Default to GPT-OSS for text config instead of Qwen2
        if text_config is None:
            text_config = {"model_type": "gpt_oss"}
        elif isinstance(text_config, dict) and "model_type" not in text_config:
            text_config["model_type"] = "gpt_oss"

        super().__init__(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
            image_seq_length=image_seq_length,
            downsample_ratio=downsample_ratio,
            projector_hidden_act=projector_hidden_act,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )


class GovMultiModalProjector(InternVLMultiModalProjector):
    """Uses InternVL's multi-modal projector unchanged"""

    pass


class GovModelOutputWithPast(InternVLModelOutputWithPast):
    """Uses InternVL's output structure unchanged"""

    pass


class GovVisionPreTrainedModel(InternVLVisionPreTrainedModel):
    pass


class GovVisionModel(InternVLVisionModel):
    pass


class GovPreTrainedModel(InternVLPreTrainedModel):
    pass


class GovModel(InternVLModel):
    """
    GOV Model that replaces InternVL's Qwen2 language model with GPT-OSS.
    """

    pass


class GovCausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    """Uses InternVL's causal LM output structure unchanged"""

    pass


class GovForConditionalGeneration(InternVLForConditionalGeneration):
    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("path/to/gov-model")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "path/to/gov-model", torch_dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://example.com/image.jpg",
        ...             },
        ...             {"type": "text", "text": "What do you see in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```
        """
        super().forward(**super_kwargs)


__all__ = [
    "GovConfig",
    "GovVisionConfig",
    "GovVisionPreTrainedModel",
    "GovVisionModel",
    "GovPreTrainedModel",
    "GovModel",
    "GovForConditionalGeneration",
]
