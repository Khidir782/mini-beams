# ==============================================================================
# BEAM -> QWEN DISTILLATION: DEVOPS TOOL ROUTER
# Execution: Run this in a Google Colab T4 GPU Runtime
# ==============================================================================

# 1. Install Dependencies
!pip install -U unsloth trl peft datasets openai pydantic

import torch
import json
import os
from google.colab import userdata
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# 2. Configuration
# Make sure to add HF_TOKEN to Colab Secrets!
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

STUDENT_SYSTEM_PROMPT = """You are a specialized DevOps & Database tool-calling agent.
Available tools:
- query_metrics(service: str, metric: "cpu"|"memory"|"latency_p99"|"error_rate", window_mins: int)
- search_logs(service: str, severity: "ERROR"|"WARN"|"FATAL", keyword: str, limit: int)
- execute_readonly_sql(database: str, sql_query: str, timeout_sec: int)
- trigger_rollback(service: str, target_version: str, notify_oncall: bool)

Respond in this exact format:
<think>
1-sentence reasoning
</think>
{"name": "tool_name", "arguments": {...}}"""

# 3. Load Student Model (Unsloth 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=1024,
    load_in_4bit=True,
)

# 4. Load the Distilled Data (Upload train_beam_tools.jsonl first!)
import json
train_data = []
with open("train_beam_tools.jsonl", "r") as f:
    for line in f:
        train_data.append(json.loads(line))

def format_row(row):
    assistant_reply = f"<think>\n{row['thought']}\n</think>\n{json.dumps(row['tool_call'])}"
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": row["user_query"]},
        {"role": "assistant", "content": assistant_reply}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

train_dataset = Dataset.from_list([format_row(r) for r in train_data])

# 5. Fine-Tune with LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        output_dir="qwen3.5-0.8b-beam-tool-router",
        report_to="none",
    ),
)

trainer.train()

# 6. Evaluation (Upload test_beam_tools.jsonl first!)
def evaluate_student(model, test_list):
    FastLanguageModel.for_inference(model)
    correct_tool, valid_json_count, exact_arg_keys = 0, 0, 0

    for item in test_list:
        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": item["user_query"]}
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(input_ids=inputs, max_new_tokens=150, temperature=0.01)
        reply = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

        json_part = reply.split("</think>")[-1].strip()
        try:
            start, end = json_part.find("{"), json_part.rfind("}") + 1
            pred = json.loads(json_part[start:end])
            valid_json_count += 1
            gold = item["tool_call"]
            if pred.get("name") == gold["name"]:
                correct_tool += 1
                if set(pred.get("arguments", {}).keys()) == set(gold["arguments"].keys()):
                    exact_arg_keys += 1
        except Exception:
            pass

    total = len(test_list)
    return {
        "Valid JSON %": round(valid_json_count / total * 100, 1),
        "Correct Tool %": round(correct_tool / total * 100, 1),
        "Exact Schema Match %": round(exact_arg_keys / total * 100, 1),
    }

test_data = []
with open("test_beam_tools.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

metrics = evaluate_student(model, test_data)
print("FINAL DISTILLATION METRICS:", metrics)