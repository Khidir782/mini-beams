import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "models/mini-beam-init"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} for training...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    mini_beam = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # 1. Load dataset
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train[:100]")

    # 2. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 3. Configure SFTTrainer
    training_args = SFTConfig(
        output_dir="./models/mini-beam-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=20,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        max_length=512,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=mini_beam,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # 4. Merge and Save
    print("Merging LoRA adapters...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("./models/mini-beam-merged")
    tokenizer.save_pretrained("./models/mini-beam-merged")
    print("Post-trained Mini-Beam saved to ./models/mini-beam-merged!")

if __name__ == "__main__":
    main()