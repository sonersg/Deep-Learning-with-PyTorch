from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

# --- Model & Quantization ---
model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True,  # Phi-3 requires this
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- LoRA Config ---
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Optimized for Phi-3
)

# --- Dataset (replace with your own) ---
# Format: {"text": "### Instruction:\n...\n### Response:\n..."}
dataset = load_dataset("json", data_files="data.jsonl", split="train")

# --- Training Args ---
training_args = TrainingArguments(
    output_dir="./phi3-qlora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # effective batch = 4
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    fp16=False,  # bfloat16 used via bnb
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,  # Reduce if OOM
)

# --- Train ---
trainer.train()
trainer.save_model("./phi3-qlora-finetuned")
