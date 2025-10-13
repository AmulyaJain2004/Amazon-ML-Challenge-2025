# ---------------------------------------------------------
# Evaluation Script for BiLSTM + Attention + Structured Features
# Generates submission.csv with sample_id and predicted price
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import os

# ----------------------------
# Setup
# ----------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ----------------------------
# Load Training Data (for feature schema & scaler)
# ----------------------------
train_df = pd.read_csv(r"dataset/train.csv")
train_df = train_df[["catalog_content", "price"]].dropna()

# Outlier handling & log transform
Q1, Q99 = train_df["price"].quantile(0.01), train_df["price"].quantile(0.99)
train_df = train_df[train_df["price"].between(Q1, Q99)]
train_df["price"] = np.log1p(np.clip(train_df["price"], 1, None))

# Extract numeric and categorical features
train_df["weight"] = train_df["catalog_content"].str.extract(r"Value:\s*([\d.]+)").astype(float)
train_df["unit"] = train_df["catalog_content"].str.extract(r"Unit:\s*(\w+)").fillna("Unknown")
train_df["pack_size"] = train_df["catalog_content"].str.extract(r"Pack of (\d+)").fillna(1).astype(float)
train_df["weight"].fillna(train_df["weight"].median(), inplace=True)
train_df["pack_size"].fillna(1, inplace=True)
train_df = pd.get_dummies(train_df, columns=["unit"], drop_first=True)

numeric_features = train_df.drop(columns=["catalog_content", "price"]).columns.tolist()

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text.strip()

train_df["catalog_content"] = train_df["catalog_content"].apply(clean_text)

# Tokenize and build vocab
words = Counter()
for sentence in train_df["catalog_content"]:
    words.update(sentence.split())

vocab_size = 12000
vocab = {word: idx + 2 for idx, (word, _) in enumerate(words.most_common(vocab_size))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_text(text, vocab, maxlen=150):
    tokens = [vocab.get(word, 1) for word in text.split()]
    tokens = tokens[:maxlen] + [0] * (maxlen - len(tokens))
    return tokens

# Scale numeric features
scaler = RobustScaler()
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])

# ----------------------------
# Load Test Data
# ----------------------------
test_df = pd.read_csv(r"dataset/test.csv")

# Ensure sample_id exists (if not, auto-create one)
if "sample_id" not in test_df.columns:
    test_df.insert(0, "sample_id", np.arange(1, len(test_df) + 1))
    print("⚠️ 'sample_id' not found in test.csv — added sequential IDs.")

# Feature engineering identical to train
test_df["weight"] = test_df["catalog_content"].str.extract(r"Value:\s*([\d.]+)").astype(float)
test_df["unit"] = test_df["catalog_content"].str.extract(r"Unit:\s*(\w+)").fillna("Unknown")
test_df["pack_size"] = test_df["catalog_content"].str.extract(r"Pack of (\d+)").fillna(1).astype(float)
test_df["weight"].fillna(train_df["weight"].median(), inplace=True)
test_df["pack_size"].fillna(1, inplace=True)
test_df = pd.get_dummies(test_df, columns=["unit"], drop_first=True)

# Match training schema
for col in train_df.columns:
    if col.startswith("unit_") and col not in test_df.columns:
        test_df[col] = 0

# Ensure correct column order
test_df = test_df.reindex(columns=["sample_id"] + numeric_features + ["catalog_content"], fill_value=0)

# Clean + encode + scale
test_df["catalog_content"] = test_df["catalog_content"].astype(str).apply(clean_text)
test_df["encoded"] = test_df["catalog_content"].apply(lambda x: encode_text(x, vocab, maxlen=150))
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# ----------------------------
# Dataset
# ----------------------------
class TestDataset(Dataset):
    def __init__(self, df, numeric_cols):
        self.X_text = torch.tensor(df["encoded"].tolist(), dtype=torch.long)
        self.X_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X_text)

    def __getitem__(self, idx):
        return self.X_text[idx], self.X_num[idx]

test_dl = DataLoader(TestDataset(test_df, numeric_features), batch_size=64)

# ----------------------------
# Model Definition (same as training)
# ----------------------------
class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, numeric_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + numeric_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_text, x_num):
        emb = self.embedding(x_text)
        lstm_out, _ = self.lstm(emb)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        combined = torch.cat([context, x_num], dim=1)
        return self.fc(self.dropout(combined)).squeeze(1)

# ----------------------------
# Load Model
# ----------------------------
model_path = r"C:\Users\hp\OneDrive\Desktop\Amazon-ML\Amazon-ML-Challenge-2025\best_model.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

model = AttentionBiLSTM(len(vocab), embed_dim=128, lstm_hidden=128, numeric_size=len(numeric_features)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Generate Predictions
# ----------------------------
preds = []
with torch.no_grad():
    for X_text, X_num in tqdm(test_dl, desc="Predicting on test set"):
        X_text, X_num = X_text.to(device), X_num.to(device)
        outputs = model(X_text, X_num)
        preds.extend(np.expm1(outputs.cpu().numpy()))  # reverse log1p transform

# ----------------------------
# Save Submission
# ----------------------------
submission = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "price": preds[: len(test_df)]  # ensure same length
})

submission.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")
