from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import joblib
import os

# === 1. Load dataset ===
df = pd.read_csv("", usecols=[1]) #add dataset to predict
texts = df.iloc[:, 0].dropna().tolist()

# === 2. Load model and tokenizer ===
model_path = "" #path to the model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# === 3. Load label encoder to get original label order ===
label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path)
labels = label_encoder.classes_  # ensures correct label mapping


# === 4. Prediction function for Contributor, Collaborator and Leader===
def predict_label(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()  # use softmax for multi-class

    top_idx = torch.tensor(probs).argmax().item()
    predicted_label = labels[top_idx]
    confidence = probs[top_idx]

    return predicted_label, confidence, probs


# === 5. Predict and collect results ===
results = []

for i, text in enumerate(texts):
    label, confidence, prob_vector = predict_label(text)
    print(f"{i + 1}. {text}\n→ Predicted: {label} (Confidence: {confidence:.2f})\n")
    results.append({
        "headline": text,
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "probabilities": prob_vector
    })

# === 6. Save results to CSV ===
output_path = "results/dread_predictions.csv"
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n Results saved to {output_path}")
