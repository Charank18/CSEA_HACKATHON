# -*- coding: utf-8 -*-
"""Copy of Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M6t6xv8Iat3O4GskiebY09B8NLCfhnyB
"""

import numpy as np
import pandas as pd

df=pd.read_csv('final_labels.csv')

df.head()

df.columns

import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Ensure stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')


stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""  # Handle missing values in text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = nltk.word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Load dataset (assuming df is already loaded)
df['body'] = df['body'].apply(clean_text)

# Drop unnecessary columns
drop_cols = ['entry_id', 'link_id', 'parent_id', 'entry_utc', 'label_date', 'week', 'sheet_order', 'image']
df = df.drop(columns=drop_cols, errors='ignore')

# Handling categorical values if necessary
df['subreddit'] = df['subreddit'].astype('category').cat.codes
df['author'] = df['author'].astype('category').cat.codes
df['split'] = df['split'].astype('category')

# Convert labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(df['level_1'].unique())}
df['labels'] = df['level_1'].map(label_mapping)

# Splitting dataset for training
df_train = df[df['split'] == 'train']
df_test = df[df['split'] == 'test']

!pip install datasets

from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["body"], padding=True, truncation=True, max_length=512)

train_dataset = Dataset.from_pandas(df_train[['body', 'labels']])
test_dataset = Dataset.from_pandas(df_test[['body', 'labels']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load transformer model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_mapping))

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./misogyny_model")
tokenizer.save_pretrained("./misogyny_model")

print("Model training complete. Saved trained model.")

import torch
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load Word2Vec for joke detection
word2vec_model = api.load("word2vec-google-news-300")
stop_words = set(stopwords.words('english'))

# Load pre-trained sarcasm and toxicity models
sarcasm_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sarcasm")
sarcasm_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sarcasm")

toxicity_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
toxicity_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def detect_sarcasm(text):
    """Detect sarcasm using a pre-trained model"""
    inputs = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = sarcasm_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "sarcasm" if prediction == 1 else "not_sarcasm"

def detect_toxicity(text):
    """Detect toxicity/harmful intent using a BERT-based model"""
    inputs = toxicity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = toxicity_model(**inputs)
    prediction = torch.sigmoid(outputs.logits).tolist()[0]
    return "harmful_intent" if any(p > 0.5 for p in prediction) else "not_harmful"

def detect_joke(text):
    """Detect jokes using sentiment analysis and word embeddings"""
    sentiment = TextBlob(text).sentiment.polarity
    joke_words = ["funny", "hilarious", "joke", "lol", "lmao", "rofl"]
    words = text.lower().split()
    humor_score = sum(1 for w in words if any(word2vec_model.similarity(w, jw) > 0.5 for jw in joke_words if w in word2vec_model))
    return "joke" if sentiment > 0.5 or humor_score > 1 else "not_joke"

def detect_context(text):
    """Classify text into sarcasm, joke, harmful intent, misogyny, or non-misogyny"""
    if detect_sarcasm(text) == "sarcasm":
        return "sarcasm"
    elif detect_joke(text) == "joke":
        return "joke"
    elif detect_toxicity(text) == "harmful_intent":
        return "harmful_intent"
    return "misogyny" if "misogyny" in text.lower() else "non-misogyny"

# Load dataset (assuming df is loaded)
df["body"] = df["body"].apply(clean_text)
df["context_label"] = df["body"].apply(detect_context)

# Map context labels to numbers
label_mapping = {"misogyny": 0, "non-misogyny": 1, "sarcasm": 2, "joke": 3, "harmful_intent": 4}
df["labels"] = df["context_label"].map(label_mapping)

# Split dataset
df_train = df[df["split"] == "train"]
df_test = df[df["split"] == "test"]

# Tokenization for BERT training
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["body"], padding="max_length", truncation=True, max_length=512)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(df_train[["body", "labels"]])
test_dataset = Dataset.from_pandas(df_test[["body", "labels"]])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load Transformer model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_mapping))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increased epochs for better generalization
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("./misogyny_model")
tokenizer.save_pretrained("./misogyny_model")

print("Training complete. Model saved.")

# Moderation Tools

def flag_content(text):
    """Flag content based on detected context."""
    context = detect_context(text)
    if context == "harmful_intent":
        return "FLAGGED: This content contains harmful intent."
    elif context == "misogyny":
        return "FLAGGED: This content contains misogynistic language."
    elif context == "sarcasm":
        return "FLAGGED: This content may contain sarcasm."
    elif context == "joke":
        return "FLAGGED: This content may be a joke."
    else:
        return "This content is clean."


def warn_user(text):
    """Warn the user based on flagged content."""
    context = detect_context(text)
    if context in ["harmful_intent", "misogyny"]:
        return "WARNING: Your content has been flagged for inappropriate language. Please review our community guidelines."
    elif context in ["sarcasm", "joke"]:
        return "WARNING: Your content may be misinterpreted. Please ensure it aligns with our community guidelines."
    else:
        return "Your content is appropriate."


def educate_user(text):
    """Provide educational feedback to the user."""
    context = detect_context(text)
    if context == "harmful_intent":
        return "EDUCATION: Harmful intent can hurt others. Learn more about positive communication here: [link]"
    elif context == "misogyny":
        return "EDUCATION: Misogynistic language perpetuates gender inequality. Learn more about inclusive language here: [link]"
    elif context == "sarcasm":
        return "EDUCATION: Sarcasm can be misunderstood. Tips on clear communication: [link]"
    elif context == "joke":
        return "EDUCATION: Humor is great, but ensure it's appropriate. Guidelines on humor in communication: [link]"
    else:
        return "Thank you for your positive contribution!"


def user_feedback(text, user_response):
    """Handle user feedback or appeals."""
    context = detect_context(text)
    if user_response == "appeal":
        return "Your appeal has been received. Our team will review your content shortly."
    elif user_response == "feedback":
        return "Thank you for your feedback. We will use it to improve our moderation system."
    else:
        return "Thank you for your input."


def log_moderation_action(text, flag_message, warn_message, educate_message):
    """Log moderation actions for review."""
    with open("moderation_log.txt", "a") as log_file:
        log_file.write(f"Text: {text}\nFlag: {flag_message}\nWarn: {warn_message}\nEducate: {educate_message}\n\n")


def moderate_content(text):
    """Moderate content by flagging, warning, and educating."""
    flag_message = flag_content(text)
    warn_message = warn_user(text)
    educate_message = educate_user(text)

    print(flag_message)
    print(warn_message)
    print(educate_message)

    # Log the moderation action
    log_moderation_action(text, flag_message, warn_message, educate_message)


def update_moderation_system(new_data):
    """Update the moderation system with new data."""
    df = pd.DataFrame(new_data)
    df["body"] = df["body"].apply(clean_text)
    df["context_label"] = df["body"].apply(detect_context)
    df["labels"] = df["context_label"].map(label_mapping)

    train_dataset = Dataset.from_pandas(df[["body", "labels"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    trainer.train(train_dataset)
    model.save_pretrained("./updated_misogyny_model")
    tokenizer.save_pretrained("./updated_misogyny_model")

    print("Moderation system updated with new data.")


# Example Usage

# Example text to moderate
example_text = "This is a joke, but it might be offensive to some people."

# Moderate the content
moderate_content(example_text)

# Example user feedback
user_response = "appeal"  # or "feedback"
feedback_message = user_feedback(example_text, user_response)
print(feedback_message)

# Example of updating the moderation system with new data
new_data = [
    {"body": "This is a harmless comment.", "split": "train"},
    {"body": "This is a misogynistic comment.", "split": "train"},
    {"body": "This is a sarcastic comment.", "split": "train"},
    {"body": "This is a harmful comment.", "split": "train"},
    {"body": "This is a joke.", "split": "train"},
]

update_moderation_system(new_data)

#TESTING

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model(model_path="./misogyny_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def predict(text, tokenizer, model, label_mapping):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    predicted_label_idx = np.argmax(probabilities)
    predicted_label = [label for label, idx in label_mapping.items() if idx == predicted_label_idx][0]
    return predicted_label, probabilities

# Load model and tokenizer
tokenizer, model = load_model()

# Define label mapping (ensure this matches the training phase)
label_mapping = {"non-misogynistic": 0, "misogynistic": 1}  # Update with actual mappings

# Example test input
text = "Women should stay in the kitchen."

# Predict
predicted_label, probabilities = predict(text, tokenizer, model, label_mapping)
print(f"Predicted Label: {predicted_label}")
print(f"Probabilities: {probabilities}")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model(model_path="./misogyny_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def predict(text, tokenizer, model, label_mapping):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    predicted_label_idx = np.argmax(probabilities)
    predicted_label = [label for label, idx in label_mapping.items() if idx == predicted_label_idx][0]
    return predicted_label, probabilities

# Load model and tokenizer
tokenizer, model = load_model()

# Define label mapping (ensure this matches the training phase)
label_mapping = {"non-misogynistic": 0, "misogynistic": 1}  # Update with actual mappings

# Example test input
text = "that women is good"

# Predict
predicted_label, probabilities = predict(text, tokenizer, model, label_mapping)
print(f"Predicted Label: {predicted_label}")
print(f"Probabilities: {probabilities}")

!pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="misogyny_model", private=False)

# ✅ Step 1: Upload ZIP file from Desktop
from google.colab import files

# This will open a file picker to select your ZIP file
uploaded = files.upload()

# ✅ Step 2: Get the file name (it will automatically detect)
import os

# Extract the uploaded file name
file_name = list(uploaded.keys())[0]
file_path = f"/content/{file_name}"

print(f"File '{file_name}' uploaded successfully!")

# ✅ Step 3: Install HuggingFace Hub library
!pip install huggingface_hub

# ✅ Step 4: Upload the ZIP file to HuggingFace
from huggingface_hub import HfApi

# Initialize HuggingFace API
api = HfApi()

# ✅ Step 5: Provide HuggingFace Repo Details
your_username = "Charankarnati18"  # 🔥 Replace this with your HuggingFace username
repo_name = "misogyny_model"     # 🔥 Replace this with your HuggingFace repo name

# ✅ Step 6: Upload ZIP to HuggingFace
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=file_name,
    repo_id=f"{your_username}/{repo_name}"
)

print(f"✅ File '{file_name}' uploaded successfully to HuggingFace!")
print(f"🚀 Access it here: https://huggingface.co/{your_username}/{repo_name}")

from transformers import pipeline

# Load the model from Hugging Face
classifier = pipeline('text-classification', model='Charankarnati18/misogyny_model')

# Provide input
input_text = "Your input text here"

# Get the model's prediction
result = classifier(input_text)
print(result)

import zipfile
import os

# Path to the .zip file
zip_path = "misogyny_model.zip"

# Directory to extract the files to
extract_dir = "misogyny_model_extracted"

# Create the directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the .zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Model files extracted to: {extract_dir}")

from transformers import AutoModel, AutoTokenizer

# Path to the extracted model directory
model_dir = "misogyny_model_extracted"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Preprocess the input
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")

# Run inference
outputs = model(**inputs)
print(outputs)

from transformers import AutoModel, AutoTokenizer

# Path to the extracted model directory
model_dir = "misogyny_model_extracted"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Preprocess the input
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")

# Run inference
outputs = model(**inputs)
print(outputs)

from huggingface_hub import upload_folder

# Path to the folder you want to upload
folder_path = "misogyny_model_extracted"

# Upload the folder to the repository
upload_folder(
    folder_path=folder_path,
    repo_id="Charankarnati18/misogyny_model",  # Repository name
    repo_type="model",  # Type of repository (model, dataset, etc.)
)

from transformers import AutoModel, AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("Charankarnati18/misogyny_model")
# Replace with your model's repository name
model_name = "Charankarnati18/misogyny_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input text
input_text = "women is good"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Example: For a classification model, get the predicted class
predicted_class = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted class: {predicted_class}")

