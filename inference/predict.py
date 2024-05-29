import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from huggingface_hub import hf_hub_download
from config import HUGGING_FACE_TOKEN, MODEL_PATH, CHECKPOINT_PATH, CACHE_DIR, MODEL_NAME, DEVICE_MAP, REPO_ID
from peft import PeftModel, LoraConfig

def download_checkpoint(repo_id, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        hf_hub_download(repo_id=repo_id, filename="adapter_model.bin", local_dir=checkpoint_dir, local_dir_use_symlinks=False, token=HUGGING_FACE_TOKEN)
        hf_hub_download(repo_id=repo_id, filename="adapter_config.json", local_dir=checkpoint_dir, local_dir_use_symlinks=False, token=HUGGING_FACE_TOKEN)


def load_model_and_tokenizer(model_path, checkpoint_path, cache_dir, device_map):
    if os.path.exists(model_path):
        print("Loading final model from:", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True, 
            use_cache=True, 
            return_dict=True, 
            torch_dtype=torch.float16,
            device_map=device_map
        )
    else:
        print("Final model not found. Loading base model and applying LoRA adapters.")
        download_checkpoint(REPO_ID, checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            cache_dir=cache_dir, 
            torch_dtype=torch.float16,
            device_map=device_map,
            token=HUGGING_FACE_TOKEN
        )
        # adapter_config = LoraConfig.from_pretrained(checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        model.save_pretrained(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(MODEL_PATH)
    return model, tokenizer

def generate_topic(text, model, tokenizer):
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

    Based on the information about the topic above, please create a short label of this topic. Make sure you only return the label and nothing more.
    [/INST] Environmental impacts of eating meat
    """
    main_prompt = f"""
    [INST]
    I have a topic that contains the following text: {text}

    Based on the information about the topic above, please create a short label of this topic. Make sure you only return the label of this text and nothing more.
    [/INST]
    """

    full_prompt = system_prompt + example_prompt + main_prompt

    gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
    result = gen(full_prompt)
    return result[0]['generated_text']

def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, CHECKPOINT_PATH, CACHE_DIR, DEVICE_MAP)
    
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = input("Enter the text or the path to the text file: ")
        if os.path.isfile(user_prompt):
            with open(user_prompt, 'r') as file:
                user_prompt = file.read()

    label = generate_topic(user_prompt, model, tokenizer)
    print("Generated Topic:", label)

if __name__ == "__main__":
    main()
