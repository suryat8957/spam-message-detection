# 📩 Spam Message Detection (ML-Based)

A machine learning project that classifies SMS messages as **spam** or **ham** using text preprocessing, TF-IDF feature extraction, and a Multinomial Naive Bayes classifier.  
This project also includes an **interactive spam checker** that allows users to classify custom messages in real-time through the terminal.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Future Enhancements](#future-enhancements)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

---

## 📖 Overview

This project trains a machine learning model to detect whether a message is **spam** (unwanted advertisements, fraud messages) or **ham** (legitimate messages).  
It uses:

- Regex-based text cleaning  
- TF-IDF vectorizer  
- Multinomial Naive Bayes  
- Train/Test splitting with stratification  
- Detailed evaluation metrics  

The script also includes an **interactive input mode** to test custom messages.

---

## ✨ Features

- Clean & preprocess raw SMS messages  
- Text vectorization using **TF-IDF**  
- Highly accurate **Multinomial Naive Bayes** classifier  
- Interactive spam detection via terminal  
- 98.6% test accuracy  
- Full classification report  
- Easily extendable for real-time filtering  

---

## 📊 Dataset

Dataset used:

**SMS Spam Collection Dataset**  
Source (Kaggle):  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Dataset statistics:

- **5572 messages total**
- **747 spam**
- **4825 ham**

Columns used:
- `label` — spam or ham  
- `text` — SMS message content  

---

## 🧠 Model Details

### 🔧 Pipeline

Pipeline([

('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),

('nb', MultinomialNB(alpha=0.1))

])


### 🧹 Text Cleaning
Performed using the clean_text() function:

- Lowercasing
- Removing URLs
- Removing email addresses
- Removing special characters
- Normalizing whitespace

### 🔍 Train/Test Split
- 80% training
- 20% testing
- Stratified
- random_state=42

---

## 📁 Project Structure
spam-message-detection/

├── spam_message.py # Main ML script + interactive checker

├── spam.csv # Dataset

├── Requirements.txt # Listed dependencies

├── README.md # Project documentation

└── .gitignore


---

## 🚀 Installation

1. Clone the repository
    ```
    git clone https://github.com/suryat8957/spam-message-detection.git
    cd spam-message-detection
    ```
2. Install required packages
    ```
    pip install -r Requirements.txt
    ```
3. Run the program
    ```
    python spam_message.py
    ```

---

## ▶️ Usage

### Interactive Spam Checker
When running the script:
--- Interactive Spam Checker ---
Type a message or 'quit' to exit

Enter message > Free entry in a prize draw!!
=> SPAM (spam probability: 0.9823)


### Predict inside Python

label, prob = predict_message("Congratulations! You have won a reward")
print(label, prob)


---

## 📈 Evaluation

Performance on test data:

| Metric              | Score |
|---------------------|-------|
| Accuracy            | 0.986 |
| Precision (Spam)    | 0.99  |
| Recall (Spam)       | 0.90  |
| F1-Score (Spam)     | 0.94  |
| Weighted Avg F1     | 0.99  |

**Summary**
- Ham detection: ~100% accuracy
- Spam detection: 90% recall
- Very strong overall performance

---

## 🔮 Future Enhancements

Planned additions:
- Real-time SMS/email/WhatsApp filtering
- Deploy as a web API (FastAPI / Flask)
- Web UI for message testing
- More ML models (Logistic Regression, SVM, BERT)
- Export trained model for production use

---

## 📦 Requirements

As listed in Requirements.txt (example):

pandas
scikit-learn

(Actual versions depend on your file.)

---

## 📜 License

This project is open-source.  
You may add a license file such as MIT if needed.

---

## 👤 Author

Surya  
GitHub: https://github.com/suryat8957
