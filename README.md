# Data Mining ECEN 758 Final Project Group 16

## Group Members
- Akanksha Shah (UIN - 136005001)
- Arvinder Singh Mundra (UIN - 335007465)
- Kyren Liu (UIN - 830004917)
- Tasfin Mahmud (UIN - 437004953)

## Project Dataset
Yelp Full Review Dataset - https://huggingface.co/datasets/Yelp/yelp_review_full

## Project Overview
The goal of this project is to develop and evaluate deep learning models for predicting Yelp review star ratings (0–4) using only the review text. Specifically, we implement and compare two architectures, a standard Bidirection Long Short-Term Memory (LSTM) network and a hybrid CNN + BiLSTM model, to understand how sequential recurrent models differ from convolution-enhanced sequence models in capturing local patterns, contextual information, and semantic structure in user reviews. Both models are trained and tested on the Yelp Review Full dataset, and their performance is assessed using metrics such as accuracy, precision, recall, and F1-score to determine which architecture better handles the complexities of multi-class text classification.

## Environment Setup

### Python Version

Verify Python3 version using command 
```
    python3 --version
```
We recommend `Python 3.12.12` version, however, 3.9/3.10/3.11 versions also usually work.


### Install Required Packages

Execute the below command to install required python packages.

```
pip3 install numpy pandas matplotlib seaborn tqdm torch datasets tokenizers optuna wordcloud scikit-learn spacy
```
or

```
pip install numpy pandas matplotlib seaborn tqdm torch datasets tokenizers optuna wordcloud scikit-learn spacy
```
Install SpaCy English Model:

```
python3 -m spacy download en_core_web_sm
```
Or

```
python -m spacy download en_core_web_sm
```

## Pre Trained Models

This project implements two deep learning architectures for Yelp review rating prediction: BiLSTM and CNN-BiLSTM. Both models were fully trained on the Yelp Review Full training dataset, and the final best-performing weights for each model were saved and stored as:

`best_lstm_yelp_model.pth` : BiLSTM Model

`best_cnn_lstm_yelp_model.pth` : CNN-BiLSTM Model

These pretrained weights are loaded during evaluation to generate predictions, confusion matrices, and performance metrics on the test dataset.

> Note: Please ensure that both .pth files are placed in the same directory as the test_script.py file. The script expects to load these checkpoints from the current working directory.

## Test Script

Execute the following command to run the test script

```
python3 test_script.py
```
Or

```
python test_script.py
```

> Note: During evaluation, the script generates confusion matrix heatmaps for both models. These plots are displayed in a separate popup window using matplotlib. Since plt.show() is a blocking operation, the script will pause at this point until the window is manually closed.
Please close each confusion matrix window to allow the script to continue executing the remaining evaluation steps.













