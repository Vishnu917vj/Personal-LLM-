# Personal LLM Fine-Tuning: Full Fine-Tuning vs SFT vs LoRA vs QLoRA

This project explores different techniques for fine-tuning a Large Language Model (LLM) on a small personal Question-Answer dataset. The objective was to understand how different fine-tuning strategies perform when teaching a language model personal information while keeping the base model, dataset, and hyperparameters as consistent as possible.

## Project Overview

Base Model:

* **Qwen2.5-0.5B-Instruct**

Fine-tuning methods explored:

* Full Fine-Tuning (Transformers Trainer)
* Full Fine-Tuning using TRL SFTTrainer
* LoRA (Low-Rank Adaptation)
* QLoRA (4-bit Quantized LoRA)

The goal was to compare these approaches in terms of:

* Training Loss
* Validation Loss
* Response Quality
* Memory Efficiency
* Ease of Deployment

---

## Dataset

The dataset consists of **116 manually created Question-Answer pairs** containing personal information about me.

Examples include:

* Name
* Family details
* Education
* Skills
* Interests
* Programming experience
* Favorite tools
* Career information

Since the dataset was manually written, it contained grammatical inconsistencies and spelling mistakes.

To improve its quality, I built a separate **dataset cleaning pipeline** using free LLMs available through OpenRouter.

The repository includes the script used for cleaning the dataset.

Dataset:

* Hugging Face Dataset ([link](https://huggingface.co/datasets/vishnuamarapu/vishnu-Qa-Assiatant-Dataset))

---

## Experiment 1 — Full Fine-Tuning (Transformers Trainer)

This experiment fine-tunes every parameter of the Qwen2.5-0.5B-Instruct model.

Final Metrics

| Epoch | Training Loss | Validation Loss |
| ----- | ------------: | --------------: |
| 19    |      0.020649 |        0.021227 |
| 20    |      0.020469 |        0.021152 |

Although the model learned the dataset well, response quality was still inconsistent for some unseen phrasings.

---

## Experiment 2 — Full Fine-Tuning using SFTTrainer

Instead of the standard Hugging Face Trainer, this experiment uses the TRL SFTTrainer, which is specifically designed for supervised instruction tuning.

Final Metrics

| Epoch | Training Loss | Validation Loss |  Entropy | Mean Token Accuracy |
| ----- | ------------: | --------------: | -------: | ------------------: |
| 19    |      0.076006 |        0.081307 | 0.118896 |            0.972616 |
| 20    |      0.073674 |        0.081961 | 0.118311 |            0.972616 |

Among all experiments, **SFTTrainer produced the most reliable and consistent responses**.

Although its validation loss is numerically higher than the standard Trainer, the generated responses better matched the expected conversational behavior.

---

## Experiment 3 — LoRA

Instead of updating every parameter, LoRA trains only a small number of adapter parameters.

Final Metrics

| Epoch | Training Loss | Validation Loss |  Entropy | Mean Token Accuracy |
| ----- | ------------: | --------------: | -------: | ------------------: |
| 19    |      0.321932 |        0.296050 | 0.308343 |            0.937255 |
| 20    |      0.333900 |        0.295641 | 0.307720 |            0.937255 |

Advantages:

* Very small adapter size
* Fast training
* Easy deployment

Limitations observed:

* More hallucinations than Full Fine-Tuning
* Personal facts were occasionally incorrect

---

## Experiment 4 — QLoRA

QLoRA combines 4-bit quantization with LoRA adapters, significantly reducing memory usage while retaining most of the model's performance.

Final Metrics

| Epoch | Training Loss | Validation Loss |  Entropy | Mean Token Accuracy |
| ----- | ------------: | --------------: | -------: | ------------------: |
| 17    |      0.285448 |        0.318845 | 0.322565 |            0.934902 |
| 18    |      0.346354 |        0.318435 | 0.322080 |            0.935686 |
| 19    |      0.333337 |        0.318324 | 0.322688 |            0.935686 |
| 20    |      0.351826 |        0.318193 | 0.322753 |            0.935686 |

QLoRA required significantly less GPU memory but achieved slightly lower response quality than the SFT fine-tuned model.

---

# Results Summary

| Method                        | Observation                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| Full Fine-Tuning              | Lowest validation loss but occasional inconsistencies       |
| Full Fine-Tuning + SFTTrainer | Best overall response quality and most reliable outputs     |
| LoRA                          | Fast and lightweight but produced more hallucinations       |
| QLoRA                         | Most memory efficient, with a small decrease in performance |

---

# Running the Notebook

The notebook contains all four experiments.

**Important:**

The tokenizer, dataset loading, preprocessing, and formatting cells are only defined in the **Full Fine-Tuning** section.

Before running the LoRA or QLoRA sections, make sure you execute the following cells first:

* Dataset loading
* Tokenizer initialization
* Dataset formatting
* Prompt formatting

These cells are shared across all experiments.

Also, **do not rerun the Full Fine-Tuning training cell** unless you intend to train again.

If you're using Google Colab (especially the free tier), rerunning large training cells may quickly exhaust the available disk storage.

---

# Hugging Face Models

Full Fine-Tuning

* https://huggingface.co/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct

SFTTrainer

* https://huggingface.co/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-sft

LoRA

* https://huggingface.co/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-LORA

QLoRA

* https://huggingface.co/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-QLora

---

# Hugging Face Spaces

Try each model interactively:

Full Fine-Tuning

https://huggingface.co/spaces/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct

SFT (Non-SFT Space name retained)

https://huggingface.co/spaces/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-nonsft

LoRA

https://huggingface.co/spaces/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-LORA

QLoRA

https://huggingface.co/spaces/vishnuamarapu/Full-Fine-Tuning-Qwen-2.5-0.5B-instruct-QLora

---

# Future Work

This project is the first in a larger series of LLM personalization experiments.

Planned experiments include:

* Prompt engineering comparisons
* Prefix Tuning
* Prompt Tuning
* P-Tuning v2
* Different prompt template designs
* Different dataset formatting strategies
* Retrieval-Augmented Generation (RAG) for personal knowledge
* Evaluation using automated benchmarks
* Fine-tuning larger Qwen models

---

# License

This project is released under the MIT License.

---

If you find this project useful or have suggestions for improvement, feel free to open an issue or contribute!
