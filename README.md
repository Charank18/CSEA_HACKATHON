# TASK3 - Toxic Comment Classification**
# The model can be found at: https://huggingface.co/spaces/Charankarnati18/TASK3


This repository contains the full source code, scripts, and dependencies required to fine-tune a RoBERTa model for toxic comment classification. The model classifies text into three categories: **Non-Toxic**, **Neutral**, and **Toxic**. Additionally, it can suggest positive rephrasings for toxic comments.

## Model Overview
The model is based on **RoBERTa (Robustly optimized BERT)** for sequence classification with three classes. It uses the Detoxify library to obtain toxicity scores and maps them to class labels. The fine-tuned model is uploaded to Hugging Face.

## Dataset
The dataset contains user comments with varying degrees of toxicity. The `final_labels.csv` file is used in this project. The script maps toxicity scores to three classes:

- **0:** Non-Toxic
- **1:** Neutral
- **2:** Toxic

## Dependencies
Install the required libraries using:

```shell
pip install tensorflow transformers pandas scikit-learn datasets detoxify torch
```

## Preprocessing
The dataset is preprocessed to:
- Drop missing values
- Convert text to lowercase
- Map toxicity scores to labels

The preprocessing script is already included in `actual_task3.py`.

## Training the Model
The RoBERTa model is trained using the Hugging Face `Trainer` API with the following parameters:
- **Epochs:** 3
- **Batch size:** 8
- **Learning rate:** 5e-5
- **Evaluation strategy:** Per epoch
- **Save strategy:** Per epoch

To train the model, run:

```shell
python actual_task3.py
```

The trained model will be saved in the `/content/saved_model` directory.

## Evaluating the Model
The evaluation script computes:
- **Accuracy**
- **AUC-ROC (One-vs-Rest)**
- **Confusion Matrix**
- **False Positive/Negative Rate**

The script automatically evaluates the model after training.

## Model Inference
You can use the trained model for inference using the following script:

```python
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch

# Load model
model_path = "/content/saved_model"
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Predict
comments = ["I hate you!", "Thank you for your response"]
inputs = tokenizer(comments, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
```

## Uploading the Model to Hugging Face
The trained model is uploaded to Hugging Face using:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="Charankarnati18/TASK3", repo_type="model")
api.upload_folder(
    folder_path="/content/saved_model",
    repo_id="Charankarnati18/TASK3",
    repo_type="model",
    commit_message="Uploading fine-tuned toxicity classification model"
)
```

The model can be found at: [https://huggingface.co/Charankarnati18/TASK3](https://huggingface.co/Charankarnati18/TASK3)


```

## Contact
For any issues, contact: [charankarnati18](https://huggingface.co/Charankarnati18)

