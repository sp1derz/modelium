"""
Training script for the Modelium LLM.

Fine-tunes a small language model (Qwen-1.8B or CodeT5) on conversion plan
generation tasks using curated examples.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from modelium.modelium_llm.training.dataset_schema import create_training_examples, TrainingExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModeliumLLMTrainer:
    """Trainer for the Modelium LLM conversion planner."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-1_8B",
        output_dir: str = "./models/modelium-llm",
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing trainer with model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def prepare_dataset(
        self,
        examples: List[TrainingExample],
        system_prompt_path: str = "modelium/modelium_llm/prompts/system_prompt.txt",
        user_template_path: str = "modelium/modelium_llm/prompts/user_prompt_template.txt",
    ) -> Dataset:
        """Prepare training dataset from examples."""
        
        # Load prompts
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        
        with open(user_template_path, "r") as f:
            user_template = f.read()
        
        # Format examples
        formatted_examples = []
        
        for example in examples:
            # Format user prompt
            user_prompt = user_template.format(
                model_descriptor=json.dumps(example.model_descriptor, indent=2),
                target_environment=example.target_environment,
                gpu_type=example.gpu_type,
                max_latency_ms=example.max_latency_ms,
                expected_qps=example.expected_qps,
                batch_size=example.batch_size,
                precision=example.precision,
                additional_context=example.additional_context,
            )
            
            # Format response (conversion plan)
            response = json.dumps(example.conversion_plan, indent=2)
            
            # Create chat format
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
            
            # Tokenize
            text = self._format_conversation(conversation)
            formatted_examples.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        logger.info(f"Prepared {len(dataset)} training examples")
        
        return dataset
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for training."""
        formatted = ""
        
        for message in conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_steps: int = 100,
    ) -> None:
        """Train the model."""
        
        # Tokenize dataset
        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=4096,
                padding="max_length",
            )
        
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            remove_columns=train_dataset.column_names,
            batched=True,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=10,
            warmup_steps=100,
            optim="adamw_torch",
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training complete!")


def main() -> None:
    """Main training function."""
    
    # Create training examples
    logger.info("Creating training examples...")
    examples = create_training_examples()
    
    # Initialize trainer
    trainer = ModeliumLLMTrainer(
        model_name="Qwen/Qwen-1_8B",
        output_dir="./models/modelium-llm-finetuned",
    )
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(examples)
    
    # Train
    trainer.train(
        train_dataset=dataset,
        num_epochs=3,
        batch_size=2,
        learning_rate=2e-5,
    )


if __name__ == "__main__":
    main()

