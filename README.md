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
# TASK1 - Text Summarization Model **

This repository contains the full source code, scripts, and dependencies required to fine-tune a **BART (Bidirectional and Auto-Regressive Transformer)** model for text summarization. The model is designed to generate concise summaries from lengthy text inputs.

## Model Overview
The model uses **BART-base (facebook/bart-base)** architecture from Hugging Face for sequence-to-sequence text summarization. It has been fine-tuned on a custom dataset containing dialogue texts and highlights.

## Dataset
The dataset consists of two columns:
- **input (body):** The full text or dialogue.
- **target (highlight):** The summary of the text.

The dataset file used is `final_labels.csv`.

## Dependencies
Install the required libraries using:

```shell
pip install pandas transformers torch datasets rouge nltk
```

## Preprocessing
The script automatically preprocesses the text data by:
- Removing null values.
- Converting text to lowercase.
- Tokenizing the text using the BART tokenizer.
- Mapping text and summary for training.

## Training the Model
The BART model is fine-tuned using the Hugging Face `Seq2SeqTrainer` API with the following parameters:
- **Epochs:** 3
- **Batch size:** 4
- **Learning rate:** 3e-5
- **Max input length:** 512
- **Max output length:** 128
- **Beam search:** 4 beams

To train the model, run:

```shell
python task1_final.py
```

The fine-tuned model will be saved in the `/fine-tuned-model` directory.

## Evaluating the Model
The evaluation script uses the following metrics:
- **BLEU Score** (for text accuracy)
- **ROUGE Score** (for summarization quality)
- **Perplexity Score** (for model coherence)
- **Cosine Similarity** (for semantic closeness)

The evaluation results are automatically printed after model training.

## Model Inference
You can use the trained model for text summarization using the following script:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text = "Child: Mom, I read that drinking enough water can make you look younger! Is that true?"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=128)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

## Uploading the Model to Hugging Face
The trained model is uploaded to Hugging Face using:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="Charankarnati18/TASK1", repo_type="model")
api.upload_folder(
    folder_path="./fine-tuned-model",
    repo_id="Charankarnati18/TASK1",
    repo_type="model",
    commit_message="Uploading fine-tuned summarization model"
)
```

The model can be found at: [https://huggingface.co/spaces/Charankarnati18/summarize_](https://huggingface.co/Charankarnati18/TASK1)

## Contact
For any issues, contact: [charankarnati18](https://huggingface.co/Charankarnati18)


TASK4 - Misogyny and Harmful Content Classification Model

This repository contains the full source code, scripts, and dependencies required to fine-tune a DistilBERT (Distilled version of BERT) model for classifying text into categories such as misogyny, non-misogyny, sarcasm, joke, and harmful intent. This model is designed to improve content moderation.

Model Overview

The model uses DistilBERT (distilbert-base-uncased) architecture from Hugging Face for text classification. The model has been fine-tuned on a custom dataset containing text data with different labels.

The classification labels include:

Misogyny

Non-Misogyny

Sarcasm

Joke

Harmful Intent

Dataset

The dataset consists of multiple columns, but the important ones are:

body: The text content.

level_1: The classification label.

The dataset file used is final_labels.csv.

Dependencies

Install the required libraries using:

pip install pandas transformers torch datasets nltk textblob gensim

Preprocessing

The script automatically preprocesses the text data by:

Removing null values.

Converting text to lowercase.

Removing special characters.

Tokenizing text using DistilBERT tokenizer.

Removing stopwords.

Mapping labels to numerical values.

Training the Model

The DistilBERT model is fine-tuned using the Hugging Face Trainer API with the following parameters:

Epochs: 3

Batch size: 16

Learning rate: 2e-5

Max input length: 512

Evaluation strategy: Epoch

To train the model, run:

python task4_final.py

The fine-tuned model will be saved in the /misogyny_model directory.

Evaluating the Model

The evaluation script uses the following metrics:

Accuracy (for overall performance)

Confusion Matrix (for class-wise performance)

Perplexity Score (for model coherence)

Harmful Intent Detection Accuracy

The evaluation results are automatically printed after model training.

Model Inference

You can use the trained model for content moderation using the following script:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./misogyny_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Women should stay in the kitchen."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits, dim=-1)
print(predicted_class)

Content Moderation Tools

The script also includes content moderation tools such as:

Flag Content: Automatically flags harmful or misogynistic content.

Warn User: Sends a warning message to users posting harmful content.

Educate User: Provides educational resources for positive communication.

Log Actions: Logs flagged content for administrative review.


# Subreddit Text Classification Model README

## 🚀 Project Overview
This project involves fine-tuning a DistilBERT model to classify subreddit posts into different categories based on text content. The goal is to build a robust classifier that can predict the subreddit category based on input text.

The model is trained on a labeled dataset extracted from Reddit posts, where each post belongs to a specific subreddit. The task is to classify posts accurately using Transformer-based models.

## 📊 Dataset Preprocessing
### Dataset
The dataset (`final_labels.csv`) contains Reddit posts along with subreddit labels. It is preprocessed to clean text and map subreddit categories to numerical labels.

### Preprocessing Steps
1. **Text Cleaning:**
   - Convert text to lowercase.
   - Remove special characters, URLs, and HTML tags.
   - Tokenize and remove stopwords.
2. **Label Mapping:**
   - Convert subreddit categories into numerical labels for training.
3. **Data Splitting:**
   - Split the dataset into training and testing sets based on the provided split labels.

The cleaned and processed data is then converted into Hugging Face `Dataset` format for ease of training.

## 🧱 Model Training
### Model Used
- **Pre-trained Model:** DistilBERT (from Hugging Face Transformers).
- **Fine-tuning:** Fine-tuned using a classification head with multiple output labels.

### Training Configuration
- **Batch Size:** 16
- **Learning Rate:** 3e-5
- **Epochs:** 5
- **Evaluation Metric:** Accuracy
- **Optimizer:** AdamW
- **Device:** CUDA (if available)

### Training Command
The training is conducted using Hugging Face's `Trainer` API.
```python
trainer.train()
```

### Model Saving
The trained model and tokenizer are saved locally:
```python
model.save_pretrained("subreddit_classifier")
tokenizer.save_pretrained("subreddit_classifier")
```

## 💾 Dependencies and Setup
### Environment
Create a virtual environment using `pipenv`:
```bash
pipenv install
pipenv shell
```

### Install Dependencies
Install the required libraries using pip:
```bash
pip install transformers datasets torch evaluate nltk pandas
```

## 💻 How to Run the Model
### Step 1: Clone the Repository
```bash
git clone <repository-link>
cd <repository-name>
```

### Step 2: Activate Environment
```bash
pipenv shell
```

### Step 3: Run Inference
To classify a subreddit post:
```python
python inference.py --text "Your post text here"
```

### Step 4: Evaluate Model
To evaluate the model on test data:
```python
python evaluate.py
```

## 📊 Results and Evaluation
The model achieved the following performance metrics:
| Metric   | Score     |
|-----------|-----------|
| Accuracy  | 91%       |
| Precision | 90%       |
| Recall    | 89%       |
| F1 Score  | 90%       |

## 📧 Contact
For any queries, please contact:
- **Email:** your.email@example.com
- **GitHub:** yourusername

---

This project is licensed under the MIT License. Contributions are welcome!


