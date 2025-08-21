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
"""
Processor class for GOV model.
"""

from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class GovProcessor(ProcessorMixin):
    r"""
    Constructs a GOV processor which wraps an image processor and a GPT-OSS tokenizer into a single processor.

    [`GovProcessor`] offers all the functionalities of the InternVL image processor and GPT-OSS tokenizer.
    See the [`~GovProcessor.__call__`] and [`~GovProcessor.decode`] for more information.

    Args:
        image_processor ([`AutoImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.image_token = "<image>"

        # Add image token to tokenizer if not present
        if self.tokenizer and self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.image_token]})

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the model.

        This method forwards the `text` and `kwargs` arguments to the tokenizer's
        [`~PreTrainedTokenizer.__call__`] method to encode the text.
        It also prepares the image(s) for the model using the image processor.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The text to be encoded. Can be a string, list of strings, or list of lists of strings.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`,
                    `List[np.ndarray]`, `List[torch.Tensor]`):
                The image(s) to be prepared. Can be a single image or list of images.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Padding strategy for text inputs.
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Truncation strategy for text inputs.
            max_length (`int`, *optional*):
                Maximum length of the returned sequences.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to return. Can be `"pt"` for PyTorch tensors.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
                - **input_ids** -- Token IDs to be fed to the model.
                - **attention_mask** -- Attention mask to avoid attending to padding tokens.
                - **pixel_values** -- Pixel values to be fed to the vision encoder.
        """
        if images is not None:
            # Process images with the image processor
            pixel_values = self.image_processor(images, return_tensors=return_tensors, **kwargs)["pixel_values"]
        else:
            pixel_values = None

        if text is not None:
            # Handle image tokens in text
            if isinstance(text, str):
                if images is not None and self.image_token not in text:
                    # Add image token at the beginning if not present
                    text = f"{self.image_token}\n{text}"
            elif isinstance(text, list):
                # Handle list of texts
                if images is not None:
                    text = [f"{self.image_token}\n{t}" if self.image_token not in t else t for t in text]

            # Tokenize text
            text_inputs = self.tokenizer(
                text,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_inputs = None

        # Combine outputs
        if text_inputs is not None:
            if pixel_values is not None:
                text_inputs["pixel_values"] = pixel_values
            return text_inputs
        else:
            return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's
        [`~PreTrainedTokenizer.batch_decode`] method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's
        [`~PreTrainedTokenizer.decode`] method.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Returns the list of input names expected by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
