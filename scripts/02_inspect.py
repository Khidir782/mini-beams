import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_weather(city: str, unit: str = "celsius"):
    """
    Get the current weather for a city.
    Args:
        city: The city to look up.
        unit: Temperature unit, "celsius" or "fahrenheit".
    """
    pass

def main():
    model_path = "models/mini-beam-init"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    mini_beam = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # 1. Inspect the hybrid layer structure
    print("\n--- Layer Structure ---")
    print("Layer 0 (Dense):")
    print(mini_beam.model.layers[0])
    print("\nLayer 5 (MoE):")
    print(mini_beam.model.layers[5])

    # 2. Test the chat template and response parser
    print("\n--- Response Parser Test ---")
    messages = [{"role": "user", "content": "What is the weather in Kuala Lumpur?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=[get_weather],
        reasoning_effort="high",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]
    outputs = mini_beam.generate(**inputs, max_new_tokens=32)

    # Parse the raw output
    parsed = tokenizer.parse_response(
        outputs[0][input_len:], prefix=inputs["input_ids"][0], tools=[get_weather]
    )
    print("Parsed Output Dictionary Keys:", parsed.keys())
    print("Parsed Message:", parsed)

if __name__ == "__main__":
    main()