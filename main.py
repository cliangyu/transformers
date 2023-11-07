from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import torch

from pprint import pprint

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.bfloat16)

# prepare inputs for the model
text_prompt1 = "<image>\nTesting\n"
url1 = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image1 = Image.open(requests.get(url1, stream=True).raw)

text_prompt2 = "What doesn this chart describe?\n<image>"
url2 = "https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png"
image2 = Image.open(requests.get(url2, stream=True).raw)


inputs = processor(text=text_prompt1, images=[image1], return_tensors="pt") # .to("cuda:0")
# inputs = processor(text=[text_prompt1, text_prompt2], images=[[image1], [image2]], return_tensors="pt") # .to("cuda:0")

# pprint(inputs)

# TODO(Nicolò): Update transformers/src/transformers/generation/utils.py

# autoregressively generate text
generation_output = model(inputs) # always set a large max_new_tokens for fuyu generate. errors happen if max_new_tokens is not set.
# generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)
# print(generation_text)

# interleaved_prompt = "<image>Generate a coco-style caption.\n<image>What doesn this chart describe?\n"
