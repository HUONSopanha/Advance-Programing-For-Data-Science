# APDS TP 03: Unsloth, LoRA, and Fine-Tuning

**Course:** Advance Programming for Data Science (APDS)  
**Department:** Department of Applied Mathematics and Statistics (AMS), Institute of Technology of Cambodia (ITC)  
**Lecturer:** Mr. VANN Visal  
**Student:** Heng Menghong (ID: e20220258)  
**Date:** October 27, 2026  

## 1. Overview of Unsloth

Unsloth is an open-source Python framework designed to make the training and fine-tuning of large language models (LLMs) more efficient. It works with the Hugging Face ecosystem and provides support for several popular model families, including Llama, Mistral, Gemma, and Qwen.

The main reason for using Unsloth is to make model training more practical when computing resources are limited. Its optimized implementations can reduce training time and GPU memory consumption. This makes it possible to fine-tune certain models on hardware that might otherwise be insufficient.

The main advantages include:

- **Training efficiency:** optimized operations can make model training faster.
- **Reduced memory consumption:** memory-saving techniques can lower GPU requirements.
- **Model customization:** a pretrained model can be adapted to a particular application or dataset.

The memory savings sometimes presented for Unsloth, such as a 70% reduction, should be understood as benchmark results rather than a guaranteed value. The actual memory requirement varies according to factors such as model size, numerical precision, sequence length, batch size, and GPU hardware. Similarly, faster training does not necessarily mean better model accuracy because the quality of the dataset and training configuration remain important.

## 2. Requirements for Using Unsloth

Before beginning fine-tuning, several technical requirements should be considered. A suitable Python environment and compatible versions of the required libraries are necessary.

For GPU-based training, an NVIDIA GPU with an appropriate driver and a compatible PyTorch/CUDA installation is normally required. The available GPU memory must also be sufficient for the selected model and training configuration.

Other requirements include:

- Internet access for downloading models and packages.
- Sufficient disk space for model checkpoints and datasets.
- Authentication or approval when a model requires restricted access.
- A supported Python version.
- Properly prepared and consistently formatted training data.

There is no single VRAM requirement that applies to every model. Memory consumption depends heavily on the model size, sequence length, quantization method, and training parameters. Using a smaller model or a 4-bit quantized model can significantly reduce memory requirements.

## 3. Environment Preparation

It is recommended to create a separate virtual environment for the project so that package versions do not interfere with other Python projects.

Before starting the training process, the following checks should be performed:

1. Confirm that the installed Python version is supported.
2. Verify that PyTorch is correctly installed.
3. Check whether CUDA is available.
4. Confirm that the selected model can be accessed.
5. Prepare the training dataset in a consistent format.

For an NVIDIA GPU, the following command can be used to inspect the installed driver and GPU:

```bash
nvidia-smi
```

After installing PyTorch, GPU availability can also be checked with Python:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` returns `False`, the CUDA/PyTorch environment should be corrected before attempting GPU-based fine-tuning.

## 4. Installing Unsloth and Loading a Llama 3 Model

Unsloth can be installed through `pip`. Inside the project environment, the following commands can be used:

```bash
python -m pip install --upgrade pip
python -m pip install unsloth
```

When working inside Jupyter Notebook, `%pip install unsloth` can be used so that the package is installed into the environment associated with the current notebook kernel.

After installation, a pretrained Llama 3 Instruct model can be loaded using Unsloth:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

The important parameters are:

- **`model_name`** specifies the pretrained model checkpoint.
- **`max_seq_length`** defines the maximum number of tokens processed in one sequence.
- **`dtype=None`** allows the framework to determine an appropriate computation data type.
- **`load_in_4bit=True`** loads the model using 4-bit quantization, which can considerably reduce the memory needed for the base model.

Loading a pretrained model does not mean that the model has already been fine-tuned. At this stage, the model has only been loaded and prepared for subsequent training.

### Llama 3 Chat Template

Instruction-tuned models expect conversations to follow a particular format. A chat template automatically organizes the roles of the participants and inserts the special tokens required by the model.

For example:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

messages = [
    {"role": "system", "content": "Explain programming clearly."},
    {"role": "user", "content": "What is a Python list?"},
    {"role": "assistant", "content": "A list stores an ordered collection of items."},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print(text)
```

The Llama 3 format contains special tokens that identify the beginning of the text, message headers, and the end of individual turns. Using the tokenizer's built-in template is preferable to manually constructing these tokens.

During training, the examples should contain the expected assistant response, so `add_generation_prompt=False` is appropriate. During inference, the conversation normally ends with the user's question, and `add_generation_prompt=True` can be used to indicate that the model should generate the next assistant response.

## 5. LoRA and Efficient Model Adaptation

LoRA, or **Low-Rank Adaptation**, is a parameter-efficient fine-tuning technique. Instead of updating all the parameters of a large pretrained model, the original weights are kept frozen while additional low-rank matrices are trained.

QLoRA extends this idea by combining LoRA with a quantized base model, commonly using 4-bit quantization. This combination can make fine-tuning large models considerably more memory-efficient.

A LoRA configuration can be created in Unsloth as follows:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

### Explanation of the Main Parameters

| Parameter | Purpose |
|---|---|
| `r=16` | Determines the rank of the LoRA adapter. A higher rank provides more adaptation capacity but increases trainable parameters. |
| `lora_alpha=16` | Controls the scaling of the adapter contribution. |
| `target_modules` | Specifies the model layers where LoRA adapters are inserted. |
| `lora_dropout=0` | Disables dropout within the LoRA adapters. |
| `bias="none"` | Prevents bias parameters from being trained through this configuration. |
| `use_gradient_checkpointing="unsloth"` | Reduces activation memory by recomputing selected values during backpropagation. |
| `random_state=3407` | Provides a fixed random seed to improve reproducibility. |
| `use_rslora=False` | Uses standard LoRA scaling instead of rank-stabilized LoRA. |
| `loftq_config=None` | Leaves LoftQ initialization disabled. |

The effective weight matrix in standard LoRA can be represented as:

\[
W_{\text{effective}} =
W_{\text{base}} + \frac{\alpha}{r}BA
\]

where \(W_{\text{base}}\) is the original frozen weight matrix and \(A\) and \(B\) are the smaller trainable matrices.

For example, when both \(r\) and \(\alpha\) are 16, the scaling factor \(\alpha/r\) is equal to 1. If rank-stabilized LoRA is enabled, the scaling behavior changes.

The major advantage is that only a relatively small number of additional parameters need to be trained instead of modifying the entire pretrained model.

## 6. Fine-Tuning Process

Fine-tuning refers to continuing the training of a pretrained model using a dataset designed for a particular task or domain. The goal is to make the model behave more appropriately for the intended application.

For example, a general language model can be fine-tuned using programming-related questions and high-quality explanations. After training, it may become more consistent when answering similar programming questions.

A typical fine-tuning process consists of the following stages:

1. Select an appropriate pretrained model.
2. Prepare and clean the training examples.
3. Divide the data into training and evaluation sets.
4. Apply the appropriate conversation template.
5. Load the model using an efficient configuration.
6. Add and configure LoRA adapters.
7. Set training parameters such as learning rate, batch size, and number of training steps.
8. Train the model.
9. Evaluate its performance on examples that were not used during training.
10. Save the trained adapters and tokenizer.

The model-loading and LoRA code described above only prepares the model. A dataset and training procedure are still required before any parameters are actually updated.

The quality of the training examples is particularly important. Poorly formatted or incorrect examples can teach undesirable behavior. Training for too many steps can also cause overfitting, where the model performs well on training examples but poorly on new inputs.

## 7. Fine-Tuning Compared with RAG

Fine-tuning and Retrieval-Augmented Generation (RAG) are two different approaches for adapting the behavior or knowledge available to a language model.

**Fine-tuning** changes model parameters, either directly or through adapters such as LoRA. It is particularly useful when the objective is to improve a model's behavior, response style, task performance, or output format.

**RAG**, in contrast, retrieves relevant information from an external collection and provides that information to the language model as context when answering a question. The model itself normally does not receive new weights during a standard RAG query.

| Aspect | Fine-Tuning | RAG |
|---|---|---|
| Main idea | Adapt the model using training examples | Retrieve external information for the current query |
| Model parameters | Model or adapter parameters are trained | Normally unchanged during inference |
| Best suited for | Behavior, style, task-specific responses, and formats | Grounded answers based on external documents |
| Updating information | Usually requires additional training | Documents and retrieval indexes can be updated |
| Main limitation | May overfit and does not guarantee factual recall | Depends on retrieving useful and relevant information |

RAG retrieval usually relies on a relevance mechanism such as embedding similarity. However, high retrieval similarity does not automatically mean that the retrieved information is correct or truthful.

Fine-tuning and RAG are not mutually exclusive. A system can use fine-tuning to improve how a model responds while using RAG to provide current or domain-specific information from an external knowledge base.

## 8. Conclusion

Unsloth provides an efficient approach for working with large language models, particularly when GPU memory and training time are important considerations. LoRA further reduces the computational cost of fine-tuning by training a small set of additional parameters instead of modifying the entire pretrained model.

A practical workflow therefore consists of preparing the environment, loading a suitable pretrained model, applying the appropriate chat template, configuring LoRA, training on high-quality examples, and evaluating the resulting model on unseen data.

Fine-tuning and RAG address different problems. Fine-tuning is mainly useful for changing model behavior and task performance, while RAG provides external information at inference time. In some applications, combining both techniques can provide a more effective solution.