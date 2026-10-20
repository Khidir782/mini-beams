import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    model_id = "reflection/Beam-1.0-501B-A23B"

    # 1. Load the official tokenizer and config
    print(f"Loading config and tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # 2. Scale down dimensions while keeping Beam's exact architectural ratios
    config.num_hidden_layers = 7           # 3 dense layers (0,1,2) + 4 MoE layers (3,4,5,6)
    config.hidden_size = 512               # Down from 5120
    config.intermediate_size = 1024        # Dense SwiGLU intermediate size
    config.moe_intermediate_size = 256     # MoE expert SwiGLU size (down from 2560)
    config.num_attention_heads = 12        # Down from 48 (maintains 6:1 GQA ratio)
    config.num_key_value_heads = 2         # Down from 8
    config.head_dim = 64                   # Down from 128

    # Scale MoE routing parameters dynamically
    for attr, val in [
        ("n_routed_experts", 32), ("num_local_experts", 32), ("num_experts", 32),
        ("n_group", 8), ("num_expert_groups", 8),
        ("topk_group", 4),
        ("num_experts_per_tok", 4),
    ]:
        if hasattr(config, attr):
            setattr(config, attr, val)

    # If the config explicitly lists layer types or global layer indices, truncate to 7 layers
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        config.layer_types = config.layer_types[:5] + [config.layer_types[5], config.layer_types[-1]]
    if hasattr(config, "mlp_layer_types") and isinstance(config.mlp_layer_types, list):
        config.mlp_layer_types = config.mlp_layer_types[:config.num_hidden_layers]

    # 3. Instantiate Mini-Beam
    # Use CPU if CUDA is not available for initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Instantiating Mini-Beam on {device} with {dtype}...")
    mini_beam = AutoModelForCausalLM.from_config(
        config, dtype=dtype
    ).to(device)

    total_params = mini_beam.num_parameters()
    print(f"Mini-Beam initialized! Total parameters: {total_params:,}")
    if device == "cuda":
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Save the config and tokenizer for subsequent scripts
    mini_beam.save_pretrained("models/mini-beam-init")
    tokenizer.save_pretrained("models/mini-beam-init")
    print("Model and tokenizer saved to models/mini-beam-init")

if __name__ == "__main__":
    main()