HUGGING_FACE_TOKEN = 'your_token_here'

TRAIN_SIZE = 20000
TEST_SIZE = 4000
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
CACHE_DIR = "models/"
OUTPUT_DIR = "final_dir/"
FINAL_MODEL_PATH = "final_model/"
NEW_MODEL_NAME = "llama-2-7b-finetuned"

TRAINING_ARGS = {
    "output_dir": "./results",
    "evaluation_strategy": "steps",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "save_steps": 100,
    "logging_steps": 50,
    "eval_steps": 100,
    "remove_unused_columns": False,
    "max_seq_length": 512
}

PEFT_CONFIG = {
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "r": 64,
    "bias": "none",
    "task_type": "Causal_LM"
}

WANDB_PROJECT = "llama2-topic-modeling"
WANDB_ENTITY = "your_wandb_entity"