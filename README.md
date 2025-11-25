# Yelp Full Review Classification: Deep Learning Sentiment Analysis

[![Project Status](https://img.shields.io/badge/Status-Complete-success)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

A comprehensive deep learning project implementing and comparing **BiLSTM** and **CNN-BiLSTM** architectures for fine-grained sentiment classification on the Yelp Review Full dataset. This repository contains both the research implementation and a professional project website showcasing our findings.

## 🎯 Project Overview

This project addresses the challenge of predicting fine-grained star ratings (0-4) from user-written Yelp reviews using state-of-the-art deep learning techniques. We developed and evaluated two neural architectures under identical experimental conditions to understand the trade-offs between model complexity and performance.

### Key Highlights

- **Dataset**: 700,000 Yelp reviews (650K training, 50K testing)
- **Task**: 5-class sentiment classification (ratings 0-4)
- **Models**: BiLSTM baseline vs. CNN-BiLSTM hybrid architecture
- **Best Performance**: 67.62% test accuracy (BiLSTM)
- **Balanced Dataset**: 130,000 reviews per rating class

## 📊 Research Findings

### Model Performance Comparison

| Model | Test Accuracy | Key Strength |
|-------|--------------|--------------|
| **BiLSTM** | **67.62%** | Better consistency across mid-range ratings |
| **CNN-BiLSTM** | 66.68% | Faster initial convergence, local pattern detection |

### Key Insights

1. **Sequence Length Matters**: Increasing max sequence length from 128 to 650 tokens significantly improved accuracy, reflecting the distributed nature of sentiment cues in reviews.

2. **Vocabulary Sweet Spot**: 20K vocabulary size provided optimal balance between coverage and sparsity. Larger vocabularies (40K) underperformed due to rare token embeddings.

3. **Architecture Complexity**: The CNN-BiLSTM hybrid showed faster early-stage learning but did not outperform the simpler BiLSTM baseline, suggesting that added architectural complexity doesn't guarantee better results for this task.

4. **Error Analysis**: 91.5% of misclassifications were off by only one star rating, indicating the models learned sentiment direction but struggled with fine-grained distinctions.

5. **Class-Specific Performance**: Both models excelled at extreme ratings (0 and 4) but faced challenges with mid-range ratings (1-3) where sentiment is more ambiguous.

## 🏗️ Architecture Details

### BiLSTM Model
```
Input (650 tokens) 
    → Embedding (128-dim) 
    → BiLSTM (2 layers, 64 hidden units per direction)
    → Dropout (0.2)
    → Fully Connected (5 classes)
```

**Specifications:**
- Embedding Dimension: 128
- Hidden Units: 64 (per direction)
- Layers: 2 (bidirectional)
- Dropout: 0.2
- Optimizer: Adam (lr=2e-3, weight decay=1e-5)

### CNN-BiLSTM Model
```
Input (650 tokens)
    → Embedding (64-dim)
    → Conv1D (64 filters, kernel=3)
    → MaxPool (stride=2)
    → BiLSTM (2 layers, 128 hidden units per direction)
    → Dropout (0.3)
    → Fully Connected (5 classes)
```

**Specifications:**
- Embedding Dimension: 64
- CNN Filters: 64 (kernel size 3)
- Hidden Units: 128 (per direction)
- Layers: 2 (bidirectional)
- Dropout: 0.3
- Optimizer: Adam (lr=5e-4, weight decay=1e-5)

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Text Cleaning**
   - Lowercasing and whitespace normalization
   - URL replacement with `<URL>` token
   - Emoticon conversion to `<POS_EMOTICON>` / `<NEG_EMOTICON>`
   - Preservation of contractions and negations

2. **Vocabulary Construction**
   - 20,000 most frequent words
   - Minimum token frequency: 5
   - Special tokens: `<PAD>`, `<UNK>`

3. **Sequence Processing**
   - Fixed length: 650 tokens (covers 99% of reviews)
   - Padding for shorter sequences
   - Truncation for longer sequences

### Hyperparameter Optimization

We employed **Optuna** for systematic hyperparameter tuning across:
- Embedding dimensions: {64, 128, 256}
- Hidden dimensions: {64, 128, 256}
- LSTM layers: {1, 2}
- Dropout rates: {0.1, 0.2, 0.3, 0.5}
- Learning rates: {5e-4, 1e-3, 2e-3}
- Batch sizes: {32, 64, 128}

### Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: Class-specific prediction accuracy
- **Recall**: Class-specific detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error pattern analysis

## 📈 Experimental Results

### Performance by Rating Class

| Rating | BiLSTM F1 | CNN-BiLSTM F1 | Difficulty |
|--------|-----------|---------------|------------|
| 0 (Worst) | **0.7875** | 0.7773 | Low |
| 1 (Poor) | **0.6307** | 0.6266 | Medium |
| 2 (Average) | **0.6121** | 0.6012 | High |
| 3 (Good) | **0.6128** | 0.5780 | High |
| 4 (Best) | 0.7398 | **0.7444** | Low |

### Training Characteristics

- **BiLSTM**: Steady, consistent learning with minimal overfitting
- **CNN-BiLSTM**: Faster initial convergence, slightly higher validation loss at convergence
- **Convergence**: Both models stabilized within 10 epochs

## 🌐 Project Website

This repository includes a professional, responsive website showcasing the complete project analysis, visualizations, and findings.

### Website Features

- **Modern Design**: Clean, professional light theme with responsive layout
- **Interactive Visualizations**: All EDA plots, training curves, and confusion matrices
- **Comprehensive Documentation**: Detailed methodology, results, and business insights
- **Mobile-Friendly**: Fully responsive design for all devices
- **Professional Typography**: Inter font family for optimal readability

### Website Structure

```
yelp-sentiment-analysis-website/
├── index.html              # Main website
├── styles.css              # Professional styling
├── images/                 # All visualizations
│   ├── bilstm-architecture.png
│   ├── cnn-bilstm-architecture.jpg
│   ├── bilstm-training-validation-loss.png
│   ├── confusion-matrix-bilstm.jpg
│   ├── rating-distribution.png
│   └── ... (additional visualizations)
├── report.tex              # LaTeX source for academic report
├── DEPLOYMENT_GUIDE.md     # GitHub Pages deployment instructions
└── README.md               # This file
```

## 🚀 Deployment

### GitHub Pages (Recommended)

1. **Create GitHub Repository**
   ```bash
   # Create a new public repository on GitHub
   ```

2. **Upload Files**
   - Upload all files from this directory
   - Ensure `images/` folder is included

3. **Enable GitHub Pages**
   - Go to Settings → Pages
   - Select `main` branch as source
   - Save and wait ~2 minutes

4. **Access Website**
   - Your site will be live at: `https://[username].github.io/[repo-name]/`

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 💼 Business Applications

### Real-World Use Cases

1. **Automated Review Triage**
   - Flag urgent negative reviews (predicted rating 0) for immediate response
   - Prioritize customer support based on sentiment severity

2. **Sentiment Trend Analysis**
   - Track predicted sentiment over time to identify declining service quality
   - Early warning system before average ratings drop

3. **Unrated Feedback Analysis**
   - Estimate satisfaction scores for text-only feedback
   - Quantify customer sentiment in open-ended survey responses

4. **Platform Integration**
   - DoorDash merchant portal for restaurant feedback analysis
   - Yelp business dashboard for sentiment monitoring

## 🔮 Future Directions

1. **Transformer Models**: Implement BERT or RoBERTa for better contextual understanding
2. **Ordinal Regression**: Use ordinal-aware loss functions to respect rating order
3. **Aspect-Based Sentiment Analysis (ABSA)**: Detect sentiment per aspect (Food, Service, Ambiance)
4. **Multi-Task Learning**: Simultaneously predict ratings and extract key opinion phrases
5. **Attention Mechanisms**: Add attention layers to identify influential review segments

## 👥 Team

**Group 16 - ECEN758 Data Mining and Analysis**

- **Akanksha Shah** - Texas A&M University
- **Arvinder Singh Mundra** - Texas A&M University  
- **Kyren Liu** - Texas A&M University
- **Tasfin Mahmud** - Texas A&M University

## 📚 References

1. G. Rao et al., "LSTM with sentence representations for document-level sentiment classification," *Neurocomputing*, vol. 308, pp. 49-57, 2018.

2. R. Belaroussi et al., "Polarity of Yelp reviews: a BERT-LSTM comparative study," *Big Data Cogn. Comput.*, vol. 9, no. 5, p. 140, 2025.

3. B. He et al., "Bi-directional LSTM-GRU based time series forecasting approach," *Int. J. Comput. Sci. Inf. Technol.*, vol. 3, no. 2, pp. 222-231, 2024.

## 📄 License

This project is created for academic purposes as part of a Data Mining course final project at Texas A&M University.

## 🙏 Acknowledgments

- **Dataset**: Yelp Review Full dataset from Hugging Face
- **Framework**: PyTorch for deep learning implementation
- **Optimization**: Optuna for hyperparameter tuning
- **Visualization**: Matplotlib, Seaborn for data visualization

---

**Built with ❤️ by Group 16 | Fall 2025 | Texas A&M University**
