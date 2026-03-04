import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch

# ====== Step 1: SFT 训练（语气对齐）======
def prepare_sft_dataset():
    # 从 nurse_persona.yaml 构造指令数据
    data = [
        {"input": "患者乳酸高怎么办？", "output": "看到这个结果确实会让人担心，但乳酸升高常见于脱水或感染，医生可以通过补液很快改善它。建议您去急诊评估。"},
        {"input": "我的血钾低吗？", "output": "您的血钾在正常范围，不用太担心。保持均衡饮食即可。"}
    ]
    return Dataset.from_list(data)

def sft_train():
    model_name = "Qwen/Qwen2-7B-Instruct"  # 或本地路径
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = prepare_sft_dataset()
    training_args = TrainingArguments(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        deepspeed="ds_config.json",  # 启用 DeepSpeed
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    model.save_pretrained("./sft_lora")

# ====== Step 2: DPO 训练（伦理对齐）======
def prepare_dpo_dataset():
    # 正样本：安全回答；负样本：危险回答（如“吃阿司匹林”）
    data = [
        {
            "prompt": "我头痛，该吃什么药？",
            "chosen": "头痛原因很多，建议先看医生明确诊断，不要自行用药。",
            "rejected": "你可以吃一片阿司匹林。"
        }
    ]
    return Dataset.from_list(data)

def dpo_train():
    from trl import DPOTrainer
    model = AutoModelForCausalLM.from_pretrained("./sft_lora", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)

    dataset = prepare_dpo_dataset()
    training_args = TrainingArguments(
        output_dir="./dpo_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=2,
        deepspeed="ds_config.json",  # 启用 DeepSpeed
        fp16=True,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    dpo_trainer.train()
    dpo_trainer.save_model("./final_agent")

if __name__ == "__main__":
    sft_train()
    dpo_train()