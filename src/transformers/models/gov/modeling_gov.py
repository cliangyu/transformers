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
"""PyTorch GOV model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_gov import GovConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GovConfig"


@dataclass
class GovCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Output type for GOV model with past key values and image hidden states.

    Args:
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            Hidden states of the model produced by the vision encoder after projection.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


class GovMultiModalProjector(nn.Module):
    """
    Multi-modal projector to align vision features with text embedding space.
    Projects from vision hidden size to text hidden size.
    """

    def __init__(self, config: GovConfig):
        super().__init__()
        self.config = config

        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size

        # Linear projection from vision to text space
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GovPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = GovConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GovVisionAttention", "GptOssDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


GOV_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`GovConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
"""


@add_start_docstrings(
    "The bare GOV model outputting raw hidden states without any specific head on top.",
    GOV_START_DOCSTRING,
)
class GovModel(GovPreTrainedModel):
    """
    GOV model combining InternVL vision encoder with GPT-OSS language model.
    """

    def __init__(self, config: GovConfig):
        super().__init__(config)
        self.config = config

        # Vision encoder from InternVL
        from ..internvl.modeling_internvl import InternVLVisionModel
        self.vision_tower = InternVLVisionModel(config.vision_config)

        # Multi-modal projector to align dimensions
        self.multi_modal_projector = GovMultiModalProjector(config)

        # GPT-OSS language model
        self.language_model = AutoModel.from_config(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        """
        Extract and project image features from pixel values.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width)
            vision_feature_layer: Which layer to extract features from
            vision_feature_select_strategy: How to select features ("default" or "full")

        Returns:
            Projected image features of shape (batch_size, num_patches, text_hidden_size)
        """
        vision_feature_layer = vision_feature_layer or self.config.vision_feature_layer
        vision_feature_select_strategy = vision_feature_select_strategy or self.config.vision_feature_select_strategy

        # Get vision features from the vision encoder
        if vision_feature_select_strategy == "full":
            vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            # Select all hidden states from the specified layer
            selected_features = vision_outputs.hidden_states[vision_feature_layer]
        else:
            # Default strategy - use last hidden state
            vision_outputs = self.vision_tower(pixel_values)
            selected_features = vision_outputs.last_hidden_state

        # Apply pixel shuffle downsampling if configured
        if self.config.downsample_ratio < 1.0:
            selected_features = self.pixel_shuffle(selected_features, self.config.downsample_ratio)

        # Project to text embedding space
        image_features = self.multi_modal_projector(selected_features)

        return image_features

    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = 0.5):
        """
        Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features: Input tensor of shape (batch_size, seq_len, hidden_size)
            scale_factor: Factor by which to downsample

        Returns:
            Downsampled tensor
        """
        batch_size, seq_len, hidden_size = vision_features.size()

        # Calculate spatial dimensions (assuming square patches)
        height = width = int(seq_len**0.5)

        if height * width != seq_len:
            # If not square, return as-is (handle CLS token case)
            return vision_features

        # Reshape to spatial format
        vision_features = vision_features.view(batch_size, height, width, hidden_size)

        # Downsample
        new_h = int(height * scale_factor)
        new_w = int(width * scale_factor)

        # Use adaptive pooling for downsampling
        vision_features = vision_features.permute(0, 3, 1, 2)  # (B, C, H, W)
        vision_features = nn.functional.adaptive_avg_pool2d(vision_features, (new_h, new_w))
        vision_features = vision_features.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Flatten back to sequence
        vision_features = vision_features.reshape(batch_size, new_h * new_w, hidden_size)

        return vision_features

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Merge text embeddings with image features at image token positions.

        Args:
            image_features: Projected image features
            inputs_embeds: Text input embeddings
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (merged_embeddings, merged_attention_mask)
        """
        batch_size, sequence_length = input_ids.shape
        image_token_id = self.config.image_token_id

        # Find positions of image tokens
        image_token_mask = input_ids == image_token_id
        num_image_tokens = image_token_mask.sum(dim=1)

        # Create output tensor
        # Calculate new sequence length (replace each image token with multiple feature tokens)
        image_seq_length = image_features.shape[1]
        new_seq_length = sequence_length + (image_seq_length - 1) * num_image_tokens.max().item()

        merged_embeds = torch.zeros(
            batch_size, new_seq_length, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=inputs_embeds.device)
        
        merged_attention_mask = torch.zeros(
            batch_size, new_seq_length, dtype=attention_mask.dtype, device=attention_mask.device
        )

        # Process each sample in the batch
        for batch_idx in range(batch_size):
            input_embeds = inputs_embeds[batch_idx]
            image_token_positions = torch.where(image_token_mask[batch_idx])[0]

            if len(image_token_positions) == 0:
                # No image tokens, just copy text
                merged_embeds[batch_idx, :sequence_length] = input_embeds
                merged_attention_mask[batch_idx, :sequence_length] = attention_mask[batch_idx]
            else:
                # Insert image features at image token positions
                curr_pos = 0
                output_pos = 0

                for img_pos in image_token_positions:
                    # Copy text before image token
                    if img_pos > curr_pos:
                        text_length = img_pos - curr_pos
                        merged_embeds[batch_idx, output_pos : output_pos + text_length] = input_embeds[
                            curr_pos:img_pos
                        ]
                        merged_attention_mask[batch_idx, output_pos : output_pos + text_length] = attention_mask[
                            batch_idx, curr_pos:img_pos
                        ]
                        output_pos += text_length

                    # Insert image features
                    merged_embeds[batch_idx, output_pos : output_pos + image_seq_length] = image_features[batch_idx]
                    merged_attention_mask[batch_idx, output_pos : output_pos + image_seq_length] = 1
                    output_pos += image_seq_length
                    curr_pos = img_pos + 1

                # Copy remaining text
                if curr_pos < sequence_length:
                    remaining = sequence_length - curr_pos
                    merged_embeds[batch_idx, output_pos : output_pos + remaining] = input_embeds[curr_pos:]
                    merged_attention_mask[batch_idx, output_pos : output_pos + remaining] = attention_mask[
                        batch_idx, curr_pos:
                    ]

        return merged_embeds, merged_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GovCausalLMOutputWithPast]:
        """
        Forward pass for the GOV model.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process image features if provided
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Merge image features with text embeddings if we have images
        if image_features is not None and input_ids is not None:
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask
            )

        # Forward through language model
        # GPT-OSS expects input_ids or inputs_embeds
        outputs = self.language_model(
            input_ids=None,  # We're using inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs

        return GovCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@add_start_docstrings(
    """GOV Model with a language modeling head on top for conditional generation.""",
    GOV_START_DOCSTRING,
)
class GovForConditionalGeneration(GovPreTrainedModel, GenerationMixin):
    """
    GOV model for conditional generation tasks (e.g., image captioning, visual question answering).
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GovConfig):
        super().__init__(config)
        self.model = GovModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    @add_start_docstrings_to_model_forward(GOV_START_DOCSTRING)
    @replace_return_docstrings(output_type=GovCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GovCausalLMOutputWithPast]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input token IDs.
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values of images.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Input embeddings.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling.
            past_key_values (`Cache`, *optional*):
                Pre-computed hidden states for faster generation.
            use_cache (`bool`, *optional*):
                Whether to return past key values.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput object.

        Returns:
            `GovCausalLMOutputWithPast` or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute loss
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GovCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            image_hidden_states=outputs.image_hidden_states if hasattr(outputs, "image_hidden_states") else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, pixel_values=None, **kwargs
    ):
        """
        Prepare inputs for generation step.
        """
        # Only use the last token for generation if past is available
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Don't pass pixel_values after first forward (they're already encoded)
        if past_key_values:
            pixel_values = None

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        return model_inputs
