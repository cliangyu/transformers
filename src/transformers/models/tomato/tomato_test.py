from transformers import TomatoProcessor, TomatoForCausalLM, TomatoImageProcessor, TomatoConfig
from PIL import Image
import requests
import json
# from helper_functions import write_to_file


with open('/p/scratch/ccstdl/transformers_cache/tomato-8b/config.json', 'r') as config_file:
    config_dict = json.load(config_file)


# Create a configuration object from the loaded dictionary
config = TomatoConfig.from_dict(config_dict)


# load model and processor
model_id = "/p/scratch/ccstdl/transformers_cache/tomato-8b"
processor = TomatoProcessor.from_pretrained(model_id)
# processor = TomatoProcessor(TomatoImageProcessor, LlamaTokenizerFast)
model = TomatoForCausalLM(config=config).to("cuda")
print(processor.tokenizer.__class__.__name__)



# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda")

# write_to_file("/p/scratch/ccstdl/xu17/tomato/tomato_test_code/tomato_log.txt", "tomato_test.py inputs", str(inputs))


# autoregressively generate text
generation_output = model.generate(**inputs)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
# assert generation_text == ['A blue bus parked on the side of a road.']
print(generation_text)
