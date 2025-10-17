# Modular BiLSTM Regression Model
import os, re
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Union, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Common stop words
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "of", "and", "or", "if", "is", "are", "to", "from", "by",
    "as", "be", "been", "this", "that", "it", "its", "was", "were", "so", "but", "not", "no", "do", "does", "did",
    "with", "about", "into", "out", "up", "down", "over", "under", "then", "than", "too", "very", "can", "could",
    "should", "would", "make", "like", "time", "has", "he", "have", "had", "what", "said", "each", "which", 
    "their", "will", "way", "many", "these", "she", "may", "use", "her", "new", "now", "old", "see", "him", 
    "two", "how", "go", "oil", "sit", "set", "run", "eat", "far", "sea", "eye", "bag", "job", "lot"
}

# Data Processing Utility Functions
def simple_tokenizer(text: str) -> List[str]:
    """
    Simple text tokenizer that removes punctuation and stop words
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
        
    # Convert to lowercase and remove non-alphanumeric characters
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Split and filter
    tokens = [token for token in text.split() if token and token not in STOP_WORDS and len(token) > 1]
    return tokens

def clean_text(text: str) -> str:
    """
    Clean text by tokenizing and rejoining
    """
    tokens = simple_tokenizer(text)
    return " ".join(tokens)

def clean_dataframe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns in DataFrame by handling inf and NaN values
    """
    df_clean = df.copy()
    
    # Replace inf and -inf with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_clean[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def encode_text_to_ids(text: str, vocab: Dict[str, int], max_length: int = 150) -> List[int]:
    """
    Encode text to sequence of IDs using vocabulary
    """
    tokens = simple_tokenizer(text)
    ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    
    # Truncate or pad to max_length
    if len(ids) >= max_length:
        return ids[:max_length]
    else:
        return ids + [vocab.get('<PAD>', 0)] * (max_length - len(ids))

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(np.mean(numerator / denominator) * 100.0)

# Configuration
class Config:
    SEED = 42
    MAX_LEN = 150
    VOCAB_SIZE = 12000
    EMBED_DIM = 256
    LSTM_HIDDEN = 128
    NUMERIC_HIDDEN = 128
    DENSE_POST = 256
    BATCH_SIZE = 128
    ACCUM_STEPS = 2
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-3
    MIXUP = False
    MIXUP_ALPHA = 0.2
    CHECKPOINT_PATH = "best_model.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility wrapper functions for compatibility
def encode_text(text, vocab, maxlen=Config.MAX_LEN):
    return encode_text_to_ids(text, vocab, maxlen)

def clean_df_numeric(df_in):
    return clean_dataframe_numeric(df_in)

def smape_numpy(y_pred, y_true, eps=1e-5):
    return float(calculate_smape(y_true, y_pred, eps))

def setup_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Dataset Class
class PriceDataset(Dataset):
    def __init__(self, df_text, df_num, y=None):
        self.X_text = torch.tensor(df_text["encoded"].tolist(), dtype=torch.long)
        self.X_num = torch.tensor(df_num.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.values.astype(np.float32), dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_text)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_text[idx], self.X_num[idx], self.y[idx]
        return self.X_text[idx], self.X_num[idx]

# Model Class
class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, numeric_size, numeric_hidden, dense_post_concat):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)

        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_size, numeric_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.attn_dropout = nn.Dropout(0.3)

        self.post_concat_dense = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + numeric_hidden, dense_post_concat),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(dense_post_concat, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x_text, x_num):
        emb = self.embedding(x_text)
        lstm_out, _ = self.lstm(emb)
        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(attn_scores + 1e-8, dim=1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        num_feat = self.numeric_branch(x_num)
        combined = torch.cat([context, num_feat], dim=1)
        combined = self.post_concat_dense(combined)
        out = self.fc(combined).squeeze(1)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

# Data Processing Functions
def load_and_preprocess_data():
    df = pd.read_csv("dataset/train.csv")
    df = df[["sample_id", "catalog_content", "price"]].dropna().reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    # Remove outliers
    Q1, Q99 = df["price"].quantile(0.01), df["price"].quantile(0.99)
    df = df[df["price"].between(Q1, Q99)].reset_index(drop=True)
    df["price"] = np.log1p(np.clip(df["price"].astype(float), 1.0, None))

    return df

def extract_numeric_features(df):
    weight_series = df["catalog_content"].astype(str).str.extract(r"Value:\s*([\d\.]+)")[0].astype(float)
    pack_series = df["catalog_content"].astype(str).str.extract(r"Pack of (\d+)")[0].astype(float)
    unit_series = df["catalog_content"].astype(str).str.extract(r"Unit:\s*(\w+)")[0].fillna("Unknown")

    df.loc[:, "weight"] = weight_series.fillna(weight_series.median(skipna=True))
    df.loc[:, "pack_size"] = pack_series.fillna(1.0)
    df.loc[:, "unit"] = unit_series
    df = pd.get_dummies(df, columns=["unit"], drop_first=True)
    
    return df

def build_vocab(df):
    df["catalog_content"] = df["catalog_content"].astype(str).apply(clean_text)
    words = Counter()
    for s in df["catalog_content"]:
        words.update(s.split())
    vocab = {w: i+2 for i, (w, _) in enumerate(words.most_common(Config.VOCAB_SIZE))}
    vocab["<PAD>"], vocab["<UNK>"] = 0, 1
    df["encoded"] = df["catalog_content"].apply(lambda x: encode_text(x, vocab, Config.MAX_LEN))
    return df, vocab

def prepare_embeddings(df):
    resnet_emb = pd.read_csv("dataset/embeddings_columns_train.csv")
    if "sample_id" in resnet_emb.columns:
        resnet_emb = resnet_emb.set_index("sample_id").reindex(df["sample_id"]).reset_index(drop=True)
    resnet_emb = clean_df_numeric(resnet_emb)
    return resnet_emb

def prepare_numeric_features(df, resnet_emb):
    numeric_features = df.drop(columns=["sample_id", "catalog_content", "price"]).columns.tolist()
    resnet_features = resnet_emb.columns.tolist()
    
    combined_df = pd.concat([df[numeric_features].reset_index(drop=True),
                             resnet_emb.reset_index(drop=True)], axis=1)
    combined_df = clean_df_numeric(combined_df)

    numeric_scaler = RobustScaler()
    df_num_scaled = numeric_scaler.fit_transform(combined_df)
    combined_features = numeric_features + resnet_features
    df_num_scaled_df = pd.DataFrame(df_num_scaled, columns=combined_features).reset_index(drop=True)
    
    return df_num_scaled_df, numeric_scaler, combined_features

# Training Functions
def create_data_loaders(df, df_num_scaled_df):
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=Config.SEED)
    train_num = df_num_scaled_df.iloc[train_df.index].reset_index(drop=True)
    val_num = df_num_scaled_df.iloc[val_df.index].reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dl = DataLoader(PriceDataset(train_df, train_num, train_df["price"]),
                          batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(PriceDataset(val_df, val_num, val_df["price"]),
                        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_dl, val_dl

def create_model(vocab_size, combined_features_len):
    model = AttentionBiLSTM(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        lstm_hidden=Config.LSTM_HIDDEN,
        numeric_size=combined_features_len,
        numeric_hidden=Config.NUMERIC_HIDDEN,
        dense_post_concat=Config.DENSE_POST
    ).to(Config.DEVICE)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using DataParallel with GPUs:", torch.cuda.device_count())
    
    return model

def setup_training_components(model):
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    loss_fn = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    grad_scaler = GradScaler()
    
    return optimizer, loss_fn, scheduler, grad_scaler

def load_checkpoint(model, optimizer, grad_scaler):
    best_smape = float("inf")
    start_epoch = 1
    
    if os.path.exists(Config.CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            grad_scaler.load_state_dict(checkpoint["scaler_state_dict"])
            best_smape = checkpoint.get("best_smape", float("inf"))
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resumed checkpoint {Config.CHECKPOINT_PATH} at epoch {start_epoch}")
        except Exception as e:
            print("Failed loading checkpoint:", e)
    
    return best_smape, start_epoch

def train_epoch(model, train_dl, optimizer, loss_fn, grad_scaler, epoch):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
    for step, batch in pbar:
        X_text, X_num, y = batch
        X_text = X_text.to(Config.DEVICE)
        X_num = X_num.to(Config.DEVICE)
        y = y.to(Config.DEVICE)

        if Config.MIXUP:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            idx = torch.randperm(X_text.size(0)).to(Config.DEVICE)
            X_num = lam * X_num + (1 - lam) * X_num[idx]
            y = lam * y + (1 - lam) * y[idx]

        with autocast(enabled=torch.cuda.is_available()):
            preds = model(X_text, X_num)
            loss = loss_fn(preds, y) / Config.ACCUM_STEPS

        grad_scaler.scale(loss).backward()
        train_loss += loss.item() * Config.ACCUM_STEPS

        if (step + 1) % Config.ACCUM_STEPS == 0:
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
    
    return train_loss / len(train_dl)

def validate_model(model, val_dl):
    model.eval()
    val_preds_log, val_trues_log = [], []
    val_preds_exp, val_trues_exp = [], []
    
    with torch.no_grad():
        for X_text_v, X_num_v, y_v in val_dl:
            X_text_v = X_text_v.to(Config.DEVICE)
            X_num_v = X_num_v.to(Config.DEVICE)
            y_v = y_v.to(Config.DEVICE)
            with autocast(enabled=torch.cuda.is_available()):
                preds_v = model(X_text_v, X_num_v).clamp(min=-20, max=20)
            val_preds_log.extend(preds_v.cpu().numpy())
            val_trues_log.extend(y_v.cpu().numpy())
            val_preds_exp.extend(np.expm1(preds_v.cpu().numpy()))
            val_trues_exp.extend(np.expm1(y_v.cpu().numpy()))

    val_smape = smape_numpy(np.array(val_preds_exp), np.array(val_trues_exp))
    val_mse_log = float(np.mean((np.array(val_preds_log) - np.array(val_trues_log))**2))
    
    return val_smape, val_mse_log, val_preds_log, val_trues_log

def save_checkpoint(model, optimizer, grad_scaler, epoch, best_smape):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_smape": best_smape,
        "scaler_state_dict": grad_scaler.state_dict()
    }
    torch.save(checkpoint, Config.CHECKPOINT_PATH)
    print(f"Saved checkpoint (SMAPE={best_smape:.4f})")

def train_model(model, train_dl, val_dl, optimizer, loss_fn, scheduler, grad_scaler):
    best_smape, start_epoch = load_checkpoint(model, optimizer, grad_scaler)
    
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimizer, loss_fn, grad_scaler, epoch)
        val_smape, val_mse_log, val_preds_log, val_trues_log = validate_model(model, val_dl)
        
        print(f"Epoch {epoch}: TrainLoss={train_loss:.6f} | Val SMAPE={val_smape:.4f} | Val MSE_log={val_mse_log:.6f}")
        scheduler.step(val_smape)

        if val_smape < best_smape:
            best_smape = val_smape
            save_checkpoint(model, optimizer, grad_scaler, epoch, best_smape)
    
    print("Training finished. Best Val SMAPE:", best_smape)
    return best_smape

def build_calibration(model, val_dl):
    print("Building calibration on validation predictions...")
    val_all_preds_log = []
    val_all_trues_log = []
    
    with torch.no_grad():
        for X_text_v, X_num_v, y_v in tqdm(val_dl, desc="Val for calibration"):
            X_text_v = X_text_v.to(Config.DEVICE)
            X_num_v = X_num_v.to(Config.DEVICE)
            with autocast(enabled=torch.cuda.is_available()):
                preds_v = model(X_text_v, X_num_v).clamp(min=-20, max=20)
            val_all_preds_log.extend(preds_v.cpu().numpy())
            val_all_trues_log.extend(y_v.cpu().numpy())

    lr = LinearRegression()
    X_fit = np.array(val_all_preds_log).reshape(-1, 1)
    y_fit = np.array(val_all_trues_log).reshape(-1, 1)
    lr.fit(X_fit, y_fit)
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0]) if isinstance(lr.intercept_, np.ndarray) else float(lr.intercept_)
    print(f"Calibration linear params (log-space): a={a:.6f}, b={b:.6f}")

    cal_preds_log = (a * X_fit.flatten()) + b
    cal_preds_exp = np.expm1(cal_preds_log)
    val_trues_exp = np.expm1(y_fit.flatten())
    cal_smape = smape_numpy(cal_preds_exp, val_trues_exp)
    print(f"Calibrated Val SMAPE: {cal_smape:.4f}")
    
    return a, b

def prepare_test_data(vocab, numeric_features, combined_features, numeric_scaler, df):
    print("Preparing test predictions...")
    test_df = pd.read_csv("dataset/test.csv")
    test_ids = test_df["sample_id"].copy()
    test_df["catalog_content"] = test_df["catalog_content"].astype(str)
    
    weight_series = test_df["catalog_content"].str.extract(r"Value:\s*([\d\.]+)")[0].astype(float)
    pack_series = test_df["catalog_content"].str.extract(r"Pack of (\d+)")[0].astype(float)
    unit_series = test_df["catalog_content"].str.extract(r"Unit:\s*(\w+)")[0].fillna("Unknown")
    test_df.loc[:, "weight"] = weight_series.fillna(df["weight"].median(skipna=True))
    test_df.loc[:, "pack_size"] = pack_series.fillna(1.0)
    test_df.loc[:, "unit"] = unit_series
    test_df = pd.get_dummies(test_df, columns=["unit"], drop_first=True)

    for col in numeric_features:
        if col not in test_df.columns:
            test_df[col] = (df[col].median(skipna=True) if col in df.columns else 0.0)

    test_df["catalog_content"] = test_df["catalog_content"].apply(clean_text)
    test_df["encoded"] = test_df["catalog_content"].apply(lambda x: encode_text(x, vocab, Config.MAX_LEN))

    test_resnet_emb = pd.read_csv("dataset/embeddings_columns_test.csv")
    if "sample_id" in test_resnet_emb.columns:
        test_resnet_emb = test_resnet_emb.set_index("sample_id").reindex(test_ids).reset_index(drop=True)
    test_resnet_emb = clean_df_numeric(test_resnet_emb)

    test_combined = pd.concat([test_df[numeric_features].reset_index(drop=True),
                               test_resnet_emb.reset_index(drop=True)], axis=1)
    for c in combined_features:
        if c not in test_combined.columns:
            test_combined[c] = 0.0
    test_combined = test_combined.reindex(columns=combined_features, fill_value=0.0)
    test_num_scaled_df = pd.DataFrame(numeric_scaler.transform(test_combined), columns=combined_features)

    return test_df, test_num_scaled_df, test_ids

def generate_predictions(model, test_df, test_num_scaled_df, test_ids, a, b):
    test_dl = DataLoader(PriceDataset(test_df, test_num_scaled_df), batch_size=Config.BATCH_SIZE, num_workers=2, pin_memory=True)

    try:
        sd = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        model_state = sd["model_state_dict"]
        if hasattr(model, "module"):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        print("Loaded checkpoint for inference:", Config.CHECKPOINT_PATH)
    except Exception as e:
        print("Failed to load checkpoint; using current weights:", e)

    model.eval()
    preds = []
    with torch.no_grad():
        for X_text_t, X_num_t in tqdm(test_dl, desc="Predicting"):
            X_text_t = X_text_t.to(Config.DEVICE)
            X_num_t = X_num_t.to(Config.DEVICE)
            with autocast(enabled=torch.cuda.is_available()):
                out = model(X_text_t, X_num_t).clamp(min=-20, max=20)
            out = out.cpu().numpy()
            out_cal = (a * out) + b
            preds.extend(np.expm1(out_cal))

    submission = pd.DataFrame({"sample_id": test_ids, "price": preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

def main():
    setup_seeds(Config.SEED)
    print("Device:", Config.DEVICE)
    print("GPUs:", torch.cuda.device_count())
    
    # Data preprocessing
    df = load_and_preprocess_data()
    df = extract_numeric_features(df)
    df, vocab = build_vocab(df)
    resnet_emb = prepare_embeddings(df)
    df_num_scaled_df, numeric_scaler, combined_features = prepare_numeric_features(df, resnet_emb)
    
    # Create data loaders
    train_dl, val_dl = create_data_loaders(df, df_num_scaled_df)
    
    # Create and setup model
    model = create_model(len(vocab), len(combined_features))
    optimizer, loss_fn, scheduler, grad_scaler = setup_training_components(model)
    
    # Train model
    best_smape = train_model(model, train_dl, val_dl, optimizer, loss_fn, scheduler, grad_scaler)
    
    # Build calibration
    a, b = build_calibration(model, val_dl)
    
    # Generate test predictions
    numeric_features = df.drop(columns=["sample_id", "catalog_content", "price", "encoded"]).columns.tolist()
    test_df, test_num_scaled_df, test_ids = prepare_test_data(vocab, numeric_features, combined_features, numeric_scaler, df)
    generate_predictions(model, test_df, test_num_scaled_df, test_ids, a, b)

if __name__ == "__main__":
    main()