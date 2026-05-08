# Yelp Review Sentiment Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A deep learning NLP system that predicts **Yelp review star ratings (1–5)** from raw review text. The project benchmarks a standard **BiLSTM** network against a hybrid **CNN-BiLSTM** architecture, with pre-trained model checkpoints included for immediate evaluation.

---

## Models

| Model | Architecture | Description |
|---|---|---|
| **BiLSTM** | Bidirectional LSTM | Sequential text encoding in both directions |
| **CNN-BiLSTM** | Conv layers → BiLSTM | Local n-gram feature extraction feeding into contextual LSTM |

Pre-trained checkpoints are included for both models — no training required to evaluate.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) via Hugging Face |
| Task | 5-class star rating prediction (1–5 stars) |
| Input | Raw review text only |

---

## Pipeline

1. **Data loading** — Stream Yelp Full dataset from Hugging Face `datasets`
2. **Tokenization** — SpaCy tokenizer with custom vocabulary construction
3. **Hyperparameter tuning** — Optuna optimization for both architectures
4. **Training** — Both models trained on full Yelp training split
5. **Evaluation** — Accuracy, precision, recall, F1-score + confusion matrix on test set
6. **Inference** — `test_script.py` for end-to-end evaluation with visualization

---

## Tech Stack

`Python` · `PyTorch` · `BiLSTM` · `CNN` · `Optuna` · `SpaCy` · `HuggingFace Datasets` · `Tokenizers` · `Scikit-learn`

---

## Setup

```bash
pip install numpy pandas matplotlib seaborn tqdm torch datasets tokenizers optuna wordcloud scikit-learn spacy
python -m spacy download en_core_web_sm
```

**Supported Python:** 3.9 / 3.10 / 3.11 / 3.12

---

## Running Evaluation

Place `best_lstm_yelp_model.pth` and `best_cnn_lstm_yelp_model.pth` in the same directory as `test_script.py`, then:

```bash
python test_script.py
```

> The script displays confusion matrix heatmaps via matplotlib. **Close each popup window** to allow the script to continue to the next evaluation step.

---

## Files

| File | Description |
|---|---|
| `Data_Mining_Project_Group_16.ipynb` | Full pipeline: preprocessing → training → evaluation |
| `test_script.py` | Standalone evaluation script with confusion matrix output |
| `best_lstm_yelp_model.pth` | Pre-trained BiLSTM weights |
| `best_cnn_lstm_yelp_model.pth` | Pre-trained CNN-BiLSTM weights |
| `Data_Mining_Project_Report.pdf` | Full project report with methodology and results |
| `index.html` / `styles.css` | Interactive web interface for live model inference |

---

## Web Interface

A browser-based inference UI is included. Open `index.html` directly in any browser to interact with model predictions visually — no server required.

---

## Course

**ECEN 758 — Data Mining and Analysis** · Texas A&M University · Fall 2025

---

## Author

**Arvinder Mundra** · [Portfolio](https://arvindermundraa.github.io/ArvinderMundra/) · [LinkedIn](https://www.linkedin.com/in/arvinder-mundraa) · [GitHub](https://github.com/arvindermundra1208)
