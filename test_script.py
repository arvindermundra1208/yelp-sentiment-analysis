import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datasets
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers
from collections import Counter
import torch.optim as optim
from collections import Counter, defaultdict
import re
import string
import math

# ============================================
# Configuration
# ============================================

MAX_LEN = 650
VOCAB_SIZE = 20000
MIN_FREQ = 5
LSTM_BATCH_SIZE = 64
CNN_LSTM_BATCH_SIZE = 32

LSTM_CONFIG = {
    "embed_dim": 128,      
    "hidden_dim": 64,     
    "num_layers": 2,      
    "dropout": 0.2,       
    "bidirectional": True,
}

CNN_LSTM_CONFIG = {
    "embed_dim": 64,         
    "cnn_out_channels": 64,   
    "cnn_kernel_size": 3,      
    "cnn_num_layers": 1,       
    "hidden_dim": 128,          
    "num_layers_lstm": 2,      
    "dropout": 0.3,            
    "bidirectional": True,
}

LSTM_CHECKPOINT_PATH = "best_lstm_yelp_model.pth"
CNN_LSTM_CHECKPOINT_PATH = "best_cnn_lstm_yelp_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Preprocessing
# ============================================

def eliminate_invalid_rows(df):
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().astype(bool)]
    return df

def preprocess_review(review):

  if not isinstance(review, str):
    return ""

  # Lowercase
  review = review.lower()

  # Normalize newlines and escaped quotes
  review = review.replace('\\n', ' ').replace('\n', ' ').replace('\\r', ' ').replace('\r', ' ')
  review = re.sub(r'\\"', ' ', review)
  review = re.sub(r"\\'", ' ', review)
  review = re.sub(r'\\', ' ', review)

  # Eliminate URLs
  review = re.sub(r'http\S+|www\.\S+|https\S+', ' <URL> ', review)

  # Positive emoticons: :) :-) :D =D ;D
  pos_emoticons = r'(:\)|:-\)|:\]|:D|:-D|=D|;D|;\)|:p|:-p|:P|:-P|xD|XD)'
  review = re.sub(pos_emoticons, ' <POS_EMOTICON> ', review)

  # Negative emoticons: :( :-( :/ :-/ :\ :-\
  neg_emoticons = r'(:\(|:-\(|:/|:-/|:\\|:-\\|:\[|:{|>:\()'
  review = re.sub(neg_emoticons, ' <NEG_EMOTICON> ', review)

  # Normalize punctuation patterns
  review = re.sub(r'-{2,}', ' - ', review)
  review = re.sub(r'\.{3,}', '<ELLIPSIS>', review)
  review = re.sub(r'([!?.,;:])\1+', r'\1', review)
  review = re.sub(r'(\?\!|\!\?|\?\!\?|\!\?\!)', ' ?! ', review)

  # Spacing around punctuations
  punctuation_to_space = string.punctuation.replace("'", "").replace("-", "").replace("<", "").replace(">", "").replace("_", "").replace("%", "").replace("+", "")
  review = re.sub(r'([%s])' % re.escape(punctuation_to_space), r' \1 ', review)

  review = review.replace('<ELLIPSIS>', ' ... ')

  # Normalize whitespaces
  review = re.sub(r'\s+', ' ', review).strip()

  return review

# ============================================
# Vocabulary & Tokenizer
# ============================================

def build_vocab_with_tokenizers(df,vocab_size,min_freq,text_col):

    # Generator that yields lists of tokens (pre-tokenized)
    def token_iterator():
        for text in df[text_col]:
            yield text.split()

    # Initialize WordLevel tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))

    special_tokens = ["<PAD>", "<UNK>"]

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    # Train from the streaming iterator
    tokenizer.train_from_iterator(token_iterator(), trainer)

    # Get vocab: token -> id
    word_to_int = tokenizer.get_vocab()

    print("Vocab size:", len(word_to_int))
    print("ID of <PAD>:", word_to_int.get("<PAD>"))
    print("ID of <UNK>:", word_to_int.get("<UNK>"))

    return word_to_int, tokenizer


def make_collate_fn(tokenizer):

    def collate_fn(batch):
        texts = [x[0] for x in batch]
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long)

        token_lists = [t.split() for t in texts]
        encodings = tokenizer.encode_batch(token_lists, is_pretokenized=True)
        input_ids = torch.tensor([enc.ids for enc in encodings], dtype=torch.long)

        return input_ids, labels

    return collate_fn


# ============================================
# Pytorch Dataset
# ============================================

class YelpTextDataset(Dataset):
    def __init__(self, df, text_col="cleaned_text", label_col="label"):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# ============================================
# 5. Models
# ============================================

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        pad_idx,
        num_layers=1,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (h_n, c_n) = self.lstm(embedded)

        if self.lstm.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        x = self.dropout(h_last)
        logits = self.fc(x)
        return logits


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        cnn_out_channels,
        cnn_kernel_size,
        cnn_num_layers,
        hidden_dim,
        num_classes,
        pad_idx,
        num_layers_lstm=1,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()

        self.cnn_num_layers = cnn_num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # CNN layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = embed_dim
        for _ in range(cnn_num_layers):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=cnn_out_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            )
            self.convs.append(conv)

            pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.pools.append(pool)

            in_channels = cnn_out_channels

        lstm_input_dim = cnn_out_channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers_lstm > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)      
        x = x.transpose(1, 2)             

        for conv, pool in zip(self.convs, self.pools):
            x = conv(x)
            x = F.relu(x)
            x = pool(x)

        x = x.transpose(1, 2)            

        output, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        x = self.dropout(h_last)
        logits = self.fc(x)
        return logits
    
# ============================================
# 6. Evaluation Helpers
# ============================================

def evaluate_on_test(model, test_loader, device):
    model.eval()
    preds_list = []
    labels_list = []

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            preds = logits.argmax(dim=1)

            preds_list.extend(preds.cpu().tolist())
            labels_list.extend(batch_y.cpu().tolist())

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    return accuracy, preds_list, labels_list


def get_confusion_matrix(test_labels, test_preds):
    y_true = np.array(test_labels)
    y_pred = np.array(test_preds)
    cm = confusion_matrix(y_true, y_pred)
    return cm


def visualize_cm(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name} model")
    plt.xlabel("Predicted Rating")
    plt.ylabel("True Rating")
    plt.tight_layout()
    plt.show()


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def get_metric_counts_from_confusion_matrix(cm):
    cm = np.array(cm)
    TP = np.diag(cm).astype(int)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    support = TP + FN
    K = cm.shape[0]

    index = [f"Rating {i}" for i in range(K)]

    counts_df = pd.DataFrame({
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Support": support
    }, index=index)

    return counts_df


def get_metrics_from_confusion_matrix(cm, counts_df):
    TP = counts_df["TP"].values
    FP = counts_df["FP"].values
    FN = counts_df["FN"].values
    TN = counts_df["TN"].values
    support = counts_df["Support"].values

    precision   = safe_div(TP, TP + FP)
    recall      = safe_div(TP, TP + FN)
    f1          = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(TN, TN + FP)
    accuracy    = safe_div(TP + TN, TP + TN + FP + FN)

    index = [f"Rating {i}" for i in range(len(TP))]

    metrics_df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Accuracy Per Class": accuracy,
        "Support": support
    }, index=index).round(4)

    return metrics_df

# ============================================
# Single-review prediction
# ============================================

def predict_rating(model, tokenizer, review, device):
    model.eval()

    text = preprocess_review(review)
    token_list = text.split()
    encoding = tokenizer.encode(token_list, is_pretokenized=True)

    input_ids = torch.tensor(encoding.ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        pred_class = logits.argmax(dim=1).item()

    return pred_class

# ============================================
# Main
# ============================================

if __name__ == "__main__":

    print("Loading Yelp Review Full dataset\n")    
    dataset = load_dataset("yelp_review_full")
    df_train = dataset["train"].to_pandas()
    df_test  = dataset["test"].to_pandas()

    # Drop invalid rows
    df_train = eliminate_invalid_rows(df_train)
    df_test = eliminate_invalid_rows(df_test)

    # Text Normalization
    print("Normalizing Text Reviews\n")  
    df_train["cleaned_text"] = df_train["text"].apply(preprocess_review)
    df_test["cleaned_text"]  = df_test["text"].apply(preprocess_review)

    # Train Validate Split
    print("Performing Train Validation Split\n")  
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42, shuffle=True, stratify=df_train['label'])

    print("Shape of training data: ", df_train.shape)
    print("Shape of validation data: ", df_val.shape)
    print("Shape of testing data: ", df_test.shape)

    print("\nBuilding vocabulary & tokenizer from training data")
    word_to_int, tokenizer = build_vocab_with_tokenizers(
        df_train,
        VOCAB_SIZE,
        MIN_FREQ,
        "cleaned_text",
    )

    pad_id = word_to_int["<PAD>"]

    tokenizer.enable_truncation(MAX_LEN)
    tokenizer.enable_padding(
        length=MAX_LEN,
        pad_id=pad_id,
        pad_token="<PAD>",
        direction="right",
    )

    print("\nPreparing test dataset and DataLoader")
    test_dataset = YelpTextDataset(df_test, text_col="cleaned_text", label_col="label")
    test_collate_fn = make_collate_fn(tokenizer)

    num_classes = df_train["label"].nunique()
    vocab_size = len(word_to_int)

    # ----------------------------------------
    # LSTM model
    # ----------------------------------------
    print("\n=== Evaluating BiLSTM model ===")

    lstm_test_loader = DataLoader(
        test_dataset,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=test_collate_fn,
    )

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=LSTM_CONFIG["embed_dim"],
        hidden_dim=LSTM_CONFIG["hidden_dim"],
        num_classes=num_classes,
        pad_idx=pad_id,
        num_layers=LSTM_CONFIG["num_layers"],
        bidirectional=LSTM_CONFIG["bidirectional"],
        dropout=LSTM_CONFIG["dropout"],
    ).to(device)

    lstm_ckpt = torch.load(LSTM_CHECKPOINT_PATH, map_location=device, weights_only=True)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm_model.eval()

    lstm_acc, lstm_preds, lstm_labels = evaluate_on_test(lstm_model, lstm_test_loader, device)
    print(f"\nBiLSTM Test Accuracy: {lstm_acc:.4f}")

    lstm_cm = get_confusion_matrix(lstm_labels, lstm_preds)
    lstm_counts_df = get_metric_counts_from_confusion_matrix(lstm_cm)
    lstm_metrics_df = get_metrics_from_confusion_matrix(lstm_cm, lstm_counts_df)

    print("\nBiLSTM performance metrics:\n")
    print(lstm_metrics_df)
    visualize_cm(lstm_cm, "LSTM")

    # ----------------------------------------
    # CNN-BiLSTM model
    # ----------------------------------------
    print("\n=== Evaluating CNN-BiLSTM model ===")

    cnn_lstm_test_loader = DataLoader(
        test_dataset,
        batch_size=CNN_LSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=test_collate_fn,
    )

    cnn_lstm_model = CNNLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=CNN_LSTM_CONFIG["embed_dim"],
        cnn_out_channels=CNN_LSTM_CONFIG["cnn_out_channels"],
        cnn_kernel_size=CNN_LSTM_CONFIG["cnn_kernel_size"],
        cnn_num_layers=CNN_LSTM_CONFIG["cnn_num_layers"],
        hidden_dim=CNN_LSTM_CONFIG["hidden_dim"],
        num_classes=num_classes,
        pad_idx=pad_id,
        num_layers_lstm=CNN_LSTM_CONFIG["num_layers_lstm"],
        bidirectional=CNN_LSTM_CONFIG["bidirectional"],
        dropout=CNN_LSTM_CONFIG["dropout"],
    ).to(device)

    cnn_lstm_ckpt = torch.load(CNN_LSTM_CHECKPOINT_PATH, map_location=device, weights_only=True)
    cnn_lstm_model.load_state_dict(cnn_lstm_ckpt["model_state_dict"])
    cnn_lstm_model.eval()

    cnn_lstm_acc, cnn_lstm_preds, cnn_lstm_labels = evaluate_on_test(
        cnn_lstm_model, cnn_lstm_test_loader, device
    )
    print(f"\nCNN-BiLSTM Test Accuracy: {cnn_lstm_acc:.4f}")

    cnn_lstm_cm = get_confusion_matrix(cnn_lstm_labels, cnn_lstm_preds)
    cnn_lstm_counts_df = get_metric_counts_from_confusion_matrix(cnn_lstm_cm)
    cnn_lstm_metrics_df = get_metrics_from_confusion_matrix(cnn_lstm_cm, cnn_lstm_counts_df)

    print("\nCNN-BiLSTM performance metrics:")
    print(cnn_lstm_metrics_df)
    visualize_cm(cnn_lstm_cm, "CNN-BiLSTM")


