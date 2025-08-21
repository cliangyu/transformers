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

"""Tokenization classes for GOV model."""

from typing import Dict, List, Optional, Union

from ...tokenization_utils_fast import PreTrainedTokenizerFast


class GovTokenizerFast(PreTrainedTokenizerFast):
    """
    GOV tokenizer based on GPT-OSS tokenizer with added vision special tokens.
    
    This tokenizer extends the base GPT-OSS tokenizer (o200k_base) with special tokens
    needed for vision-language processing, following the InternVL pattern.
    """
    
    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        # Vision special tokens following InternVL pattern
        self.start_image_token = "<img>"
        self.end_image_token = "</img>"
        self.context_image_token = "<IMG_CONTEXT>"
        self.video_token = "<video>"
        
        # Add vision special tokens to the tokenizer
        special_tokens_dict = {
            "additional_special_tokens": [
                self.start_image_token,
                self.end_image_token, 
                self.context_image_token,
                self.video_token,
                "<quad>",
                "</quad>",
                "<ref>",
                "</ref>",
                "<box>",
                "</box>",
            ]
        }
        
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **special_tokens_dict,
            **kwargs,
        )
        
        # Set up token IDs for the vision tokens
        self._setup_vision_token_ids()
    
    def _setup_vision_token_ids(self):
        """Set up vision token IDs after tokenizer initialization."""
        self.start_image_token_id = self.convert_tokens_to_ids(self.start_image_token)
        self.end_image_token_id = self.convert_tokens_to_ids(self.end_image_token)
        self.context_image_token_id = self.convert_tokens_to_ids(self.context_image_token)
        self.video_token_id = self.convert_tokens_to_ids(self.video_token)
    
    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())
    
    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save the tokenizer vocabulary."""
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
        
        files = self._tokenizer.model.save(save_directory, filename_prefix)
        return tuple(files)


__all__ = ["GovTokenizerFast"]