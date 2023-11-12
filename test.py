from PIL import Image
import requests

import torch

from transformers import FuyuProcessor, FuyuForCausalLM, AutoTokenizer


device = torch.device('cpu')

# Load the processor and the model

model_id = 'adept/fuyu-8b'

processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id).to(device)


def generate(text, images=None, max_new_tokens=32, verbose=False):
    inputs = processor(text=text, images=images, return_tensors='pt').to(device)

    if verbose:
        print(f'\n\n\n\n{inputs}', end='\n\n\n\n')

    generation_output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)

    return generation_text


def main():
    # Prepare inputs for the model
    
    # Different prompts, to test different cases

    text_prompt_0 = 'Q: Describe <image>\nA:'
    text_prompt_1 = 'Q: Given <image and <image>, spot the differences\nA:'
    text_prompt_2 = 'Q: Are <image>image> the same image?'
    text_prompt_3 = 'Tell me about <image>'
    text_prompt_4 = 'Q: Tell me about Germany\nA:'

    # Images urls

    image_url_0 = 'https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png'
    image_url_1 = 'https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png'

    # Images

    image_0 = Image.open(requests.get(image_url_0, stream=True).raw).convert('RGB')
    image_1 = Image.open(requests.get(image_url_1, stream=True).raw).convert('RGB')

    # Test case #1: There is a single image

    output = generate(
        text=text_prompt_0,
        images=image_0,

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #2: Two images (P.S. The current model is not tuned to
    # handle multiple images, so it will mash them up even though the
    # patch embeddings are clearly separated by other tokens)

    output = generate(
        text=text_prompt_1,
        images=[image_0, image_1],

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #3: The image placeholders are adjacent

    output = generate(
        text=text_prompt_2,
        images=[image_0, image_1],

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #4: The image is located at the end of the prompt

    output = generate(
        text=text_prompt_3,
        images=image_0,

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #5: There is no image at all

    output = generate(
        text=text_prompt_4,

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #6: There is no text at all

    output = generate(
        images=image_0,

        verbose=True
    )

    print(output, end='\n\n\n\n--------')

    # Test case #7: The number of images doesn't match the number of placeholders (this should yield an exception)

    try:
        output = generate(
            text=text_prompt_0,

            verbose=True
        )

        print(output, end='\n\n\n\n--------')
    except Exception as exception:
        print(exception, end='\n\n\n\n')
        print('Success!', end='\n\n\n\n--------')

    # Text case #8: Batching

    output = generate(
        text=[text_prompt_0, text_prompt_0],
        images=[[image_0], [image_1]],

        verbose=True
    )


if __name__ == '__main__':
    main()
