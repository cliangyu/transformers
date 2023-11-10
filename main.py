from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch

from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature

from pprint import pprint

device = 'cpu' # torch.device('cuda:0')

# load model and processor
model_id = "adept/fuyu-8b"

tokenizer = AutoTokenizer.from_pretrained(model_id)

processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)

# prepare inputs for the model
text_prompt1 = "Input: Describe the difference between <image> and <image>\nOutput:"
url1 = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image1 = Image.open(requests.get(url1, stream=True).raw)

text_prompt2 = "<image>What doesn this chart describe?\n<image>TestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTestTest<image>Other test"
url2 = "https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png"
image2 = Image.open(requests.get(url2, stream=True).raw)

inputs = FuyuBatchFeature(processor(text=text_prompt1, images=[[image1, image2]], return_tensors="pt")[0]).to(device) # .to("cuda:0")

print(inputs)

for k, v in inputs.items():
    if not isinstance(v, torch.Tensor):
        print(f'Skipping {k}')

        if isinstance(v, list):
             print(k, len(v), v[0].shape)

        continue

    print(k, v.shape)

# print(tokenizer.decode(inputs[0]['input_ids'][0].tolist()).replace('|SPEAKER|', '_').replace('|NEWLINE|', '_'))
# print(inputs[0]['image_patches_indices'][0].tolist())

# print(inputs['input_ids'].shape)

# inputs = processor(text=[text_prompt1, text_prompt2], images=[[image1], [image2]], return_tensors="pt") # .to("cuda:0")

# pprint(inputs)

# autoregressively generate text
generation_output = tokenizer.decode(model.generate(**inputs, max_new_tokens=128)[0].detach().cpu().tolist()) # always set a large max_new_tokens for fuyu generate. errors happen if max_new_tokens is not set.
print(generation_output)

# generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)
# print(generation_text)

# interleaved_prompt = "<image>Generate a coco-style caption.\n<image>What doesn this chart describe?\n"