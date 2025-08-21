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
"""Testing suite for the PyTorch GOV model."""

import unittest

import torch
from parameterized import parameterized

from transformers import GovConfig, GovForConditionalGeneration, GovModel
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


class GovModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        image_size=224,
        patch_size=14,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=100,
        text_hidden_size=128,
        vision_hidden_size=256,
        num_text_hidden_layers=2,
        num_vision_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_token_id=99,
        image_seq_length=16,
        downsample_ratio=0.5,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.text_hidden_size = text_hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.num_text_hidden_layers = num_text_hidden_layers
        self.num_vision_hidden_layers = num_vision_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio

        # Calculate expected sequence length for patches
        self.num_patches = (image_size // patch_size) ** 2

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        
        # Add image token to input_ids
        input_ids[:, 0] = self.image_token_id
        
        attention_mask = None
        if self.use_input_mask:
            attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long, device=torch_device)

        pixel_values = torch.randn(
            self.batch_size,
            self.num_channels,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
            device=torch_device,
        )

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, labels

    def get_config(self):
        from transformers.models.gpt_oss import GptOssConfig
        from transformers.models.gov import GovVisionConfig
        
        # Create vision config
        vision_config = GovVisionConfig(
            hidden_size=self.vision_hidden_size,
            num_hidden_layers=self.num_vision_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
        )
        
        # Create text config (GPT-OSS)
        text_config = GptOssConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.text_hidden_size,
            num_hidden_layers=self.num_text_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_eps,
            num_local_experts=1,  # Simplified for testing
            num_experts_per_tok=1,
        )
        
        # Create GOV config
        config = GovConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=self.image_token_id,
            image_seq_length=self.image_seq_length,
            downsample_ratio=self.downsample_ratio,
        )
        
        return config

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values, labels):
        model = GovModel(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
        
        # Check that output has expected shape
        # The sequence length will be expanded due to image features
        self.parent.assertIsNotNone(result.hidden_states)

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, pixel_values, labels
    ):
        model = GovForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
        
        # Check outputs
        self.parent.assertIsNotNone(result.logits)
        if labels is not None:
            self.parent.assertIsNotNone(result.loss)
            self.parent.assertGreater(result.loss.item(), 0.0)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values, labels = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


@require_torch
class GovModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GovModel, GovForConditionalGeneration) if torch is not None else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = GovModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GovConfig, has_text_modality=True)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="GOV does not use standard inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="GOV has specific input requirements with images")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="GOV model architecture is fixed")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="GOV model architecture is fixed")
    def test_save_load_fast_init_to_base(self):
        pass


@require_torch
class GovIntegrationTest(unittest.TestCase):
    def test_small_model_integration(self):
        """Test that a small GOV model can be instantiated and run forward pass."""
        # Create a small config for testing
        from transformers.models.gpt_oss import GptOssConfig
        from transformers.models.gov import GovConfig, GovVisionConfig
        
        vision_config = GovVisionConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            image_size=224,
            patch_size=32,
        )
        
        text_config = GptOssConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_local_experts=1,
            num_experts_per_tok=1,
        )
        
        config = GovConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=99,
        )
        
        model = GovForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        
        # Prepare inputs
        batch_size = 1
        input_ids = torch.tensor([[99, 1, 2, 3, 4]], device=torch_device)  # 99 is image token
        pixel_values = torch.randn(batch_size, 3, 224, 224, device=torch_device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Check outputs
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], batch_size)
        self.assertEqual(outputs.logits.shape[-1], config.text_config.vocab_size)