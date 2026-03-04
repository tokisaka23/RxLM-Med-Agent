import os
import argparse
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from swift import Swift, LoRAConfig, Trainer, TrainingArguments
from swift.utils import get_logger

# Professional tactical logging
logger = get_logger()

class RxLMTrainer:
    """
    Production-grade LoRA Fine-tuning Suite for RxLM-Med Agent.
    Supports multi-strategy experiments: M-Clean, M-Physics, M-Style, M-Composite.
   
    """
    def __init__(self, model_id="qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_id = model_id
        # Define experimental splits per architecture section 3.3
        self.dataset_root = "data/splits"
        self.strategy_map = {
            "M-Clean": "train_clean.jsonl",
            "M-Physics": "train_physics.jsonl",
            "M-Style": "train_style.jsonl",
            "M-Composite": "train_composite.jsonl" #
        }

    def run_mission(self, strategy, epochs=3, batch_size=2, learning_rate=1e-4):
        logger.info(f"--- INITIALIZING MISSION: [{strategy}] ---")
        
        if strategy not in self.strategy_map:
            raise ValueError(f"CRITICAL ERROR: Strategy {strategy} not found in tactical map.")

        # 1. Load Model & Tokenizer
        # Qwen2.5-VL is the backbone of Visual Layer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # 2. Apply LoRA Configuration
        lora_config = LoRAConfig(
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = Swift.prepare_model(model, lora_config)
        logger.info(f"LoRA adapters injected. Parameters optimized for {strategy}.")

        # 3. Setup Training Arguments
        train_args = TrainingArguments(
            output_dir=f"outputs/{strategy}",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="steps",
            save_steps=100,
            evaluation_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="tensorboard"
        )

        # 4. Initialize SWIFT Trainer
        dataset_path = os.path.join(self.dataset_root, self.strategy_map[strategy])
        logger.info(f"Deploying dataset: {dataset_path}")
        
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset_path, # Path to your .jsonl file
            tokenizer=tokenizer
        )

        # 5. Execute Training
        logger.info("Engaging training loop. Monitoring CER/Loss dynamics...")
        trainer.train()
        
        # 6. Secure Weights
        save_path = os.path.join(train_args.output_dir, "final_weights")
        trainer.save_model(save_path)
        logger.info(f"MISSION SUCCESS: Weights secured at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RxLM-Med Tactical Fine-tuning Suite")
    parser.add_argument("--strategy", type=str, required=True, choices=["M-Clean", "M-Physics", "M-Style", "M-Composite"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Launching tactical deployment
    tactical_unit = RxLMTrainer()
    tactical_unit.run_mission(
        strategy=args.strategy, 
        epochs=args.epochs, 
        learning_rate=args.lr
    )