import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def main():
    model_path = "./models/mini-beam-merged"
    
    # Only attempt if merged model exists
    try:
        print(f"Loading merged model from {model_path} for quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        quantized_mini_beam = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )

        print(f"4-Bit Quantized Memory Footprint: {quantized_mini_beam.get_memory_footprint() / 1e6:.2f} MB")
    except Exception as e:
        print(f"Quantization failed or model not found: {e}")
        print("Note: 4-bit quantization via bitsandbytes requires a CUDA GPU.")

if __name__ == "__main__":
    main()