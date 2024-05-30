import os
import random
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
import wandb
from config import (
    HUGGING_FACE_TOKEN, 
    TRAIN_SIZE, 
    TEST_SIZE, 
    MODEL_NAME, 
    CACHE_DIR, 
    OUTPUT_DIR, 
    FINAL_MODEL_PATH, 
    NEW_MODEL_NAME, 
    TRAINING_ARGS, 
    PEFT_CONFIG,
    WANDB_PROJECT,
    WANDB_ENTITY
)

# Initialize wandb, mlflow can also be setup here
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
    "train_size": TRAIN_SIZE,
    "test_size": TEST_SIZE,
    "model_name": MODEL_NAME,
    "training_args": TRAINING_ARGS,
    "peft_config": PEFT_CONFIG
})

# Load ML-ArXiv papers dataset from huggingface
full_dataset = load_dataset("CShorten/ML-ArXiv-Papers")

# Generate random indices for train and test sets
train_indices = random.sample(range(100000), TRAIN_SIZE)
test_indices = random.sample(range(100000,115000), TEST_SIZE)

# Select subsets based on the sampled indices
reduced_train_dataset = full_dataset["train"].select(train_indices)
reduced_test_dataset = full_dataset["train"].select(test_indices)

# Create a new DatasetDict
reduced_dataset = DatasetDict({
    "train": reduced_train_dataset,
    "test": reduced_test_dataset
})

def format_data(item):
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics.
    <</SYS>>
    """

    example_prompt = """
    I have a topic that contains the following documents:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the word food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
    [/INST] Environmental impacts of eating meat
    """

    user_prompt = item["abstract"]

    main_prompt = f"""
    [INST]
    I have a topic that contains the following text: {user_prompt}

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label of this text and nothing more.
    [/INST]
    """

    formatted_data = system_prompt + example_prompt + main_prompt
    return {"formatted_data": formatted_data}

# Apply the formatting function
formatted_dataset = reduced_dataset.map(lambda items: format_data(items))

# Use a suitable tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGING_FACE_TOKEN, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Quantization to load an LLM with less GPU memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

device_map = {"": 0}

# Load the model for text generation
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    use_auth_token=HUGGING_FACE_TOKEN,
    device_map=device_map,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Setup PEFT for LoRA adaptation
peft_config = LoraConfig(
    lora_alpha=PEFT_CONFIG["lora_alpha"],
    lora_dropout=PEFT_CONFIG["lora_dropout"],
    r=PEFT_CONFIG["r"],
    bias=PEFT_CONFIG["bias"],
    task_type=PEFT_CONFIG["task_type"]
)

lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

# Setup the training
training_args = TrainingArguments(
    output_dir=TRAINING_ARGS["output_dir"],
    evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
    learning_rate=TRAINING_ARGS["learning_rate"],
    per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
    num_train_epochs=TRAINING_ARGS["num_train_epochs"],
    save_steps=TRAINING_ARGS["save_steps"],
    logging_steps=TRAINING_ARGS["logging_steps"],
    eval_steps=TRAINING_ARGS["eval_steps"],
    remove_unused_columns=TRAINING_ARGS["remove_unused_columns"]
)

trainer = SFTTrainer(
    model=lora_model,
    args=training_args,
    peft_config=peft_config,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    dataset_text_field="formatted_data",
    max_seq_length=TRAINING_ARGS["max_seq_length"],
    tokenizer=tokenizer,
    packing=False,
    callbacks=[wandb.log]
)

trainer.train()

# Save the trained model
output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

wandb.finish()
