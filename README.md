# Topic Labeling with Hugging Face Transformers

This project demonstrates how to use a fine-tuned model to generate labels for topics. The original model from Meta is finetuned using quantization and LoRA for the topic modeling task.

## Table of Contents

- [Setup](#setup)
- [Downloading Checkpoints and Model](#downloading-checkpoints-and-model)
- [Running the Script](#running-the-script)
- [Configuration](#configuration)

## Setup

1. **Install the required dependencies**:
    ```sh
    pip install transformers torch datasets peft huggingface_hub
    ```

2. **Set your Hugging Face token as an environment variable**:
    ```sh
    export HUGGING_FACE_TOKEN='your_token_here'
    ```

## Downloading Checkpoints and Model

### Updating the Script

The script `predict.py` is designed to download the checkpoint from Hugging Face Model Hub if the final model is not available locally.

## Running the Script

You can run the script with either direct text input or by providing a path to a text file.

### With Direct Text Input

```sh
python predict.py "Your input text here"
```

### With Text File
```
python predict.py /path/to/your/textfile.txt
```

## Configuration

The configuration is managed through the config.py file. Ensure you set up the following variables:

```
HUGGING_FACE_TOKEN = 'your_token_here'
MODEL_PATH = "final_model/"
CHECKPOINT_PATH = "final_checkpoint/"
CACHE_DIR = "models/"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE_MAP = {"": 0}
```

## Running in a Docker Environment

To dockerize the prediction code using an NVIDIA Docker image, you will need to have Docker installed on your system; it installs the required dependencies and copies your code into the container. You can run the code using Docker as follows:

### Build the Docker Image:
Navigate to the inference folder and run the following command
```
docker build -t topic-labeling .
```

### Run the Docker container:
```
docker run --gpus all -e -v /path/to/your/models:/app/models topic-labeling "Your input text here"

```

### Running with a Text File input:
If you want to run the container with a text file input, you can mount a volume and provide the file path as an argument:
```
docker run --gpus all -e -v /path/to/your/models:/app/models -v /path/to/your/textfile.txt:/app/textfile.txt topic-labeling /app/textfile.txt
```
