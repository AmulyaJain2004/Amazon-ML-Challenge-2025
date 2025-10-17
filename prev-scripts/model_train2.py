# ---------------------------------------------------------
# BiLSTM + Attention + Structured Features Price Prediction
# with Log Price, SMAPE Loss, Robust Scaling, Regularization
# Optimized for fast training (≤5 epochs)
# OFFLINE VERSION (No nltk / No Internet)
# ---------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import re
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from tqdm import tqdm
    from collections import Counter

    # ----------------------------
    # Setup (Offline)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal local stopwords list
    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "for", "of", "and", "or", "if", "is", "are",
        "to", "from", "by", "as", "be", "been", "this", "that", "it", "its", "was", "were",
        "so", "but", "not", "no", "do", "does", "did", "with", "about", "into", "out", "up",
        "down", "over", "under", "then", "than", "too", "very", "can", "could", "should", "would"
    }

    def simple_tokenizer(text):
        """Offline-safe tokenizer — lowercase, remove punctuation, split by spaces."""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t and t not in STOP_WORDS]
        return tokens

    def clean_text(text):
        tokens = simple_tokenizer(text)
        return " ".join(tokens)

    # ----------------------------
    # Load Data
    # ----------------------------
    df = pd.read_csv(r"dataset/train.csv")
    df = df[["catalog_content", "price"]].dropna()

    # ----------------------------
    # Outlier Handling & Log Transform
    # ----------------------------
    Q1, Q99 = df["price"].quantile(0.01), df["price"].quantile(0.99)
    df = df[df["price"].between(Q1, Q99)]
    df["price"] = np.log1p(np.clip(df["price"], 1, None))

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    df["weight"] = df["catalog_content"].str.extract(r"Value:\s*([\d.]+)").astype(float)
    df["unit"] = df["catalog_content"].str.extract(r"Unit:\s*(\w+)").fillna("Unknown")
    df["pack_size"] = df["catalog_content"].str.extract(r"Pack of (\d+)").fillna(1).astype(float)
    df["weight"].fillna(df["weight"].median(), inplace=True)
    df["pack_size"].fillna(1, inplace=True)
    df = pd.get_dummies(df, columns=["unit"], drop_first=True)
    numeric_features = df.drop(columns=["catalog_content", "price"]).columns.tolist()

    # ----------------------------
    # Clean Text
    # ----------------------------
    df["catalog_content"] = df["catalog_content"].apply(clean_text)

    # ----------------------------
    # Tokenize & Build Vocabulary (Offline)
    # ----------------------------
    words = Counter()
    for sentence in df["catalog_content"]:
        words.update(sentence.split())

    vocab_size = 12000
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(words.most_common(vocab_size))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def encode_text(text, vocab, maxlen=150):
        tokens = simple_tokenizer(text)
        ids = [vocab.get(word, 1) for word in tokens]
        ids = ids[:maxlen] + [0] * (maxlen - len(ids))
        return ids

    df["encoded"] = df["catalog_content"].apply(lambda x: encode_text(x, vocab, maxlen=150))

    # ----------------------------
    # Scale numeric features
    # ----------------------------
    scaler = RobustScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # ----------------------------
    # Dataset
    # ----------------------------
    class PriceDataset(Dataset):
        def __init__(self, df, numeric_cols):
            self.X_text = torch.tensor(df["encoded"].tolist(), dtype=torch.long)
            self.X_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)
            self.y = torch.tensor(df["price"].values, dtype=torch.float32)

        def __len__(self):
            return len(self.X_text)

        def __getitem__(self, idx):
            return self.X_text[idx], self.X_num[idx], self.y[idx]

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    train_dl = DataLoader(PriceDataset(train_df, numeric_features), batch_size=64, shuffle=True)
    val_dl = DataLoader(PriceDataset(val_df, numeric_features), batch_size=64)

    # ----------------------------
    # Model with Attention + Regularization
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

    model = AttentionBiLSTM(
        len(vocab), embed_dim=128, lstm_hidden=128, numeric_size=len(numeric_features)
    ).to(device)

    # ----------------------------
    # SMAPE Loss + Optimizer
    # ----------------------------
    def smape_loss(y_pred, y_true):
        epsilon = 1e-6
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true) + epsilon) / 2
        return torch.mean(numerator / denominator) * 100

    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_smape = float("inf")
    patience, wait = 2, 0
    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_text, X_num, y_batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            X_text, X_num, y_batch = X_text.to(device), X_num.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_text, X_num)
            loss = smape_loss(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_text, X_num, y_batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                X_text, X_num, y_batch = X_text.to(device), X_num.to(device), y_batch.to(device)
                preds = model(X_text, X_num)
                val_preds.extend(np.expm1(preds.cpu().numpy()))
                val_true.extend(np.expm1(y_batch.cpu().numpy()))

        val_smape = np.mean(
            np.abs(np.array(val_preds) - np.array(val_true))
            / ((np.abs(np.array(val_preds)) + np.abs(np.array(val_true))) / 2)
        ) * 100

        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_dl):.4f} | Val SMAPE={val_smape:.2f}")

        if val_smape < best_smape:
            best_smape = val_smape
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print(f"\n✅ Best Validation SMAPE: {best_smape:.2f}")

    # ----------------------------
    # TEST PREDICTION AND SUBMISSION GENERATION
    # ----------------------------
    print("\n🚀 Generating predictions for test.csv...")

    test_df = pd.read_csv(r"dataset/test.csv")

    # Feature engineering identical to train
    test_df["weight"] = test_df["catalog_content"].str.extract(r"Value:\s*([\d.]+)").astype(float)
    test_df["unit"] = test_df["catalog_content"].str.extract(r"Unit:\s*(\w+)").fillna("Unknown")
    test_df["pack_size"] = test_df["catalog_content"].str.extract(r"Pack of (\d+)").fillna(1).astype(float)
    test_df["weight"].fillna(df["weight"].median(), inplace=True)
    test_df["pack_size"].fillna(1, inplace=True)
    test_df = pd.get_dummies(test_df, columns=["unit"], drop_first=True)

    # Match columns
    for col in df.columns:
        if col.startswith("unit_") and col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df.reindex(columns=numeric_features + ["catalog_content"], fill_value=0)

    # Clean + encode + scale
    test_df["catalog_content"] = test_df["catalog_content"].astype(str).apply(clean_text)
    test_df["encoded"] = test_df["catalog_content"].apply(lambda x: encode_text(x, vocab, maxlen=150))
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])

    # Test dataset
    class TestDataset(Dataset):
        def __init__(self, df, numeric_cols):
            self.X_text = torch.tensor(df["encoded"].tolist(), dtype=torch.long)
            self.X_num = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

        def __len__(self):
            return len(self.X_text)

        def __getitem__(self, idx):
            return self.X_text[idx], self.X_num[idx]

    test_dl = DataLoader(TestDataset(test_df, numeric_features), batch_size=64)

    # Load best model
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for X_text, X_num in tqdm(test_dl, desc="Predicting on test set"):
            X_text, X_num = X_text.to(device), X_num.to(device)
            outputs = model(X_text, X_num)
            preds.extend(np.expm1(outputs.cpu().numpy()))

    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": preds
    })
    submission.to_csv("submission.csv", index=False)
    print("✅ Predictions saved to submission.csv")
