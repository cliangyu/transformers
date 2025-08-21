#!/usr/bin/env python3
"""Simple test to verify GOV model components."""

import torch
from transformers.models.gov import GovConfig, GovVisionConfig
from transformers.models.gpt_oss import GptOssConfig

def test_gov_config():
    """Test GOV configuration."""
    print("Testing GOV configuration...")
    
    # Create configs
    vision_config = GovVisionConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        image_size=224,
        patch_size=32,
    )
    
    text_config = GptOssConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_local_experts=1,
        num_experts_per_tok=1,
        sliding_window=128,  # Set a sliding window value
        layer_types=["full_attention", "full_attention"],  # Use full attention only
    )
    
    config = GovConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=99,
    )
    
    print(f"✅ Config created successfully")
    print(f"  Vision hidden size: {config.vision_config.hidden_size}")
    print(f"  Text hidden size: {config.text_config.hidden_size}")
    print(f"  Image token ID: {config.image_token_id}")
    
    return config

def test_gov_components():
    """Test individual GOV components."""
    from transformers.models.gov.modeling_gov import GovMultiModalProjector, GovModel
    
    print("\nTesting GOV components...")
    
    config = test_gov_config()
    
    # Test projector
    print("\nTesting multi-modal projector...")
    projector = GovMultiModalProjector(config)
    dummy_vision_features = torch.randn(2, 10, config.vision_config.hidden_size)
    projected = projector(dummy_vision_features)
    print(f"✅ Projector output shape: {projected.shape}")
    assert projected.shape == (2, 10, config.text_config.hidden_size)
    
    # Test vision encoder separately
    print("\nTesting vision encoder...")
    from transformers.models.internvl.modeling_internvl import InternVLVisionModel
    vision_model = InternVLVisionModel(config.vision_config)
    pixel_values = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        vision_outputs = vision_model(pixel_values)
    print(f"✅ Vision encoder output shape: {vision_outputs.last_hidden_state.shape}")
    
    # Test GPT-OSS separately
    print("\nTesting GPT-OSS language model...")
    from transformers.models.gpt_oss import GptOssModel
    text_model = GptOssModel(config.text_config)
    input_ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        text_outputs = text_model(input_ids=input_ids)
    print(f"✅ GPT-OSS output shape: {text_outputs.last_hidden_state.shape}")
    
    print("\n✅ All components tested successfully!")

if __name__ == "__main__":
    try:
        test_gov_components()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)