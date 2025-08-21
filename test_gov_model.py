#!/usr/bin/env python3
"""Quick test script to verify GOV model implementation."""

import torch
from transformers.models.gov import GovConfig, GovForConditionalGeneration, GovVisionConfig
from transformers.models.gpt_oss import GptOssConfig

def test_gov_model():
    print("Testing GOV model implementation...")
    
    # Create small configs for testing
    vision_config = GovVisionConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        image_size=224,
        patch_size=32,
    )
    
    text_config = GptOssConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_local_experts=1,
        num_experts_per_tok=1,
    )
    
    config = GovConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=999,
    )
    
    print("Creating model...")
    model = GovForConditionalGeneration(config)
    model.eval()
    
    # Prepare dummy inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 999, (batch_size, seq_length))
    input_ids[:, 0] = 999  # Set first token as image token
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Expected shape: (batch_size={batch_size}, seq_length=variable, vocab_size={text_config.vocab_size})")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(
        input_ids=input_ids[:1, :2],  # Use shorter sequence for generation
        pixel_values=pixel_values[:1],
        max_length=20,
        do_sample=False,
    )
    print(f"Generated shape: {generated.shape}")
    
    print("\n✅ GOV model test passed!")
    return True

if __name__ == "__main__":
    try:
        test_gov_model()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)