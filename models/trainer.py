import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib

# === 1. Load and Clean Data ===
df = pd.read_csv() #add csv file to be read for training

# Clean label column - avoid nan in classification report
df['label'] = df['label'].astype(str).str.strip()
df = df[df['label'].notna() & (df['label'] != '')]
df = df[df['label'] != 'Reader'].reset_index(drop=True)

# === 2. Extract texts and labels ===
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# === 3. Encode labels ===
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)

# Save the label encoder for later use
label_encoder_path = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_path)

# === 4. Train-test split ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

# === 5. Tokenization ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)


# === 6. Dataset class ===
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# === 7. Load model ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# === 8. Training arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch"
)

# === 9. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# === 10. Train ===
trainer.train()

# === 11. Evaluate ===
outputs = trainer.predict(val_dataset)
preds = np.argmax(outputs.predictions, axis=1)

# === 12. Classification Report ===
# Ensure we only evaluate on valid present classes
present_labels = sorted(set(val_labels) | set(preds))
present_label_names = label_encoder.inverse_transform(present_labels)

print("\nClassification Report:\n")
print(classification_report(
    val_labels,
    preds,
    labels=present_labels,
    target_names=present_label_names,
    zero_division=0
))

# === 13. Save model and tokenizer ===
model.save_pretrained("./")
tokenizer.save_pretrained("./")
print("\n Model and tokenizer saved.")
