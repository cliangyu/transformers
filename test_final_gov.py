#!/usr/bin/env python3

import sys
sys.path.insert(0, '/data/users/leonlc/transformers/src')

print("Testing final GOV model implementation...")

try:
    from transformers.models.gov.configuration_gov import GovConfig
    from transformers.models.gov.modeling_gov import GovModel
    
    # Create minimal config for testing
    config = GovConfig(
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": [32, 32],
            "patch_size": [8, 8],
            "num_channels": 3,
        },
        text_config={
            "model_type": "gpt_oss",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    
    print(f"✅ Config created: {config.model_type}")
    print(f"   Text config type: {config.text_config.model_type}")
    
    # Test model creation
    model = GovModel(config)
    print(f"✅ GovModel initialized successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()