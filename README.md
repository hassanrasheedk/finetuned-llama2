# Topic Labeling with LLama 2

This project demonstrates how to use a fine-tuned model to generate labels for topics. The model can be loaded directly if available or loaded with LoRA adapters if the final model is not available.

## Table of Contents

- [Setup](#setup)
- [Running the Script](#running-the-script)
- [Configuration](#configuration)
- [Running in a Docker Environment](#running-in-a-docker-environment)
- [Running the Training Script](#running-the-training-script)
- [Alternative Topic Modeling Method with BERTopic](#alternative-topic-modeling-method-with-bertopic)
- [Challenges, Key Findings, and Improvements](#challenges-key-findings-and-improvements)


## Setup

1. **Install the required dependencies**:
    ```sh
    pip install transformers torch datasets peft huggingface_hub
    ```

2. **Set your Hugging Face in the config**:
    ```sh
    HUGGING_FACE_TOKEN='your_token_here'
    ```

## Running the Script

The script `predict.py` is designed to download the finetuned checkpoints from Hugging Face Model Hub (if the final model is not already available locally). You can run the script with either direct text input or by providing a path to a text file.

### With Direct Text Input

```sh
python predict.py "Your input text here"
```

### Running with a Text File

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

To dockerize the prediction code using an NVIDIA Docker image, you will need to have Docker installed on your system. This installs the required dependencies and copies your code into the container. You can run the code using Docker as follows:

### Build the Docker Image:
Navigate to the inference folder and run the following command
```
docker build -t topic-labeling .
```

### Run the Docker container:
```
docker run --gpus all topic-labeling "Your input text here"

```

### Running with a Text File input:
If you want to run the container with a text file input, you can mount a volume and provide the file path as an argument:
```
docker run --gpus all -v /path/to/your/textfile.txt:/app/textfile.txt topic-labeling /app/textfile.txt
```

## Running the Training Script

The training uses two SOTA techniques to lower the compute requirements and finetune the large 7B param model namely: **Quantization** and **Low-Rank Adaptation**.

To run the training script and fine-tune the model, follow these steps:

1. **Ensure you have all the dependencies installed**:
    ```sh
    pip install transformers torch datasets peft bitsandbytes trl wandb
    ```
2. **Set your Hugging Face token in the config**:
   ```sh
   HUGGING_FACE_TOKEN='your_token_here'
   ```
3. **Run the training script**:
   ```sh
   python train.py
   ```
This script will fine-tune the model using the specified configuration and save the final model. I used quantization and LoRA adapdation to keep the memory usage of the model low and to train less number of parameters rather than the complete model.

## Alternative Topic Modeling Method with BERTopic

If you are interested in exploring an alternative topic modeling method, you can use BERTopic. BERTopic is a topic modeling technique that leverages BERT embeddings and clustering algorithms to create dense clusters and generate meaningful topics.

For more details on BERTopic, you can check the following resources:

- **Colab Notebook**: [Explore BERTopic on Colab](https://colab.research.google.com/drive/1QCERSMUjqGetGGujdrvv_6_EeoIcd_9M?usp=sharing)
- **GitHub Repository**: [BERTopic GitHub Repo](https://github.com/MaartenGr/bertopic?tab=readme-ov-file)

These resources provide a comprehensive guide on how to use BERTopic for topic modeling tasks, including installation, usage, and examples.

## Challenges, Key Findings, and Improvements

### Main Challenges

**Handling Large Models**:
One of the primary challenges was managing the computational and memory resources required to handle large language models. This involved optimizing memory usage and ensuring that the model could be loaded and run efficiently, even on hardware with limited resources.

**Integration of LoRA Adapters**:
Integrating Low-Rank Adaptation (LoRA) adapters into the model fine-tuning process was complex. Ensuring the adapters were correctly applied and merged with the base model required careful handling and verification.

**Managing Dependencies and Environment**:
Setting up the development environment with all the necessary dependencies, particularly those related to GPU acceleration and the BitsandBytes library for quantization, presented compatibility and configuration challenges.

**Data Handling and Preprocessing**:
Ensuring that input data was correctly formatted and tokenized for the model was crucial. Handling different text inputs, whether from a file or user input, required robust preprocessing steps.


### Key Findings

**Effectiveness of LoRA**:
LoRA adapters were effective in fine-tuning the model with relatively lower computational requirements compared to traditional fine-tuning methods. This technique proved to be a practical approach for adapting the large model to our specific task.

**Importance of Cache Management**:
Proper management of the cache directory was critical. Setting a writable cache directory ensured smooth downloading and loading of model components, preventing permission errors and improving overall workflow efficiency.

**Prompt Engineering**:
Crafting effective prompts significantly impacted the quality of the generated topics. Including example prompts and system prompts helped guide the model to produce more accurate and relevant labels.


### Ideas for Improvement

**Code Quality Enhancements**:

Modularization: Break down the code into more modular components, such as separate functions or classes for model loading, preprocessing, and inference. This would improve readability and maintainability.

Error Handling: Implement more comprehensive error handling to manage exceptions and provide informative error messages, which can help in debugging and improving user experience.

**Documentation**:

Detailed Docstrings: Add detailed docstrings to all functions and classes, explaining their purpose, inputs, outputs, and any exceptions they might raise.

Usage Examples: Include usage examples in the documentation to help users understand how to run the code and interpret the results.

**Model Evaluation and Improvement**:

Quantitative Metrics: Implement quantitative evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess the performance of the model more objectively.

Hyperparameter Tuning: Explore hyperparameter tuning to optimize model performance further. This could involve experimenting with different learning rates, batch sizes, and adapter configurations.

Data Augmentation: Use data augmentation techniques to enrich the training data, potentially improving the model’s generalization capabilities.

**Deployment Considerations**:

Containerization Improvements: Ensure the Docker container is optimized for performance and security. This includes using a lightweight base image and minimizing the container size by only including necessary dependencies.

Scalability: Consider implementing scalable deployment solutions, such as using Kubernetes for orchestration and managing multiple instances of the model for load balancing.