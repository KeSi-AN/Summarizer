
# 📄 Internship Project Report

**Title:** Extractive Text Summarization Using Machine Learning  
**Intern Name:** Adham Ansari  


---

## ✅ 1. Project Overview

This project implements an **Extractive Text Summarization tool** that selects the most important sentences from long-form documents to generate concise summaries. It uses a **machine learning-based classification approach** with **TF-IDF features** and a **Logistic Regression model** to determine which sentences are most relevant to the summary.

---

## ✅ 2. Objectives

- Automatically generate high-quality summaries from long articles.
- Use an extractive technique that selects actual sentences from the document.
- Achieve a minimum of **70% accuracy** in sentence classification.
- Build a simple **GUI** for users to paste text and receive a summary.
- Ensure the model is trained with proper evaluation metrics (accuracy, precision, recall, confusion matrix).

---

## ✅ 3. Tools & Technologies Used

| Category           | Tools / Libraries                            |
|--------------------|----------------------------------------------|
| Language           | Python 3                                      |
| Libraries          | `scikit-learn`, `nltk`, `pandas`, `matplotlib`, `seaborn` |
| Feature Extraction | `TfidfVectorizer`                             |
| Model              | `LogisticRegression` from `scikit-learn`     |
| API Access         | `Kaggle API`                                  |
| GUI Framework      | `Flask`                                       |
| Notebook           | `Jupyter Notebook`                            |
| Dataset Source     | Kaggle - [Wiki Summary Dataset](https://www.kaggle.com/datasets/santhiyapremkumar/wiki-summary-dataset) |

---

## ✅ 4. Dataset Details

- **Name:** Wiki Summary Dataset (Kaggle)
- **Content:** Contains full Wikipedia articles and their corresponding human-written summaries.
- **Size:** ~20,000 samples
- **Import Method:** Downloaded directly in the training notebook using Kaggle API.

---

## ✅ 5. Methodology

1. **Data Preprocessing**
   - Documents and summaries were split into sentences using `nltk.sent_tokenize`.
   - Each sentence was labeled `1` if it appeared in the summary, `0` otherwise.

2. **Feature Engineering**
   - Used **TF-IDF Vectorization** (`max_features=5000`) to convert sentences to numerical features.

3. **Model Training**
   - Trained a **Logistic Regression** model to classify sentences as summary-worthy or not.
   - Used 80/20 train-test split.

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

5. **Model Saving**
   - The trained model and vectorizer were saved using `joblib`.

6. **GUI Development**
   - A Flask-based web app allows users to input text and receive a summary.
   - The model is loaded and applied in real-time.

---

## ✅ 6. Results

| Metric            | Value     |
|-------------------|-----------|
| Accuracy          | ~85%      |
| Precision (avg)   | ~0.84     |
| Recall (avg)      | ~0.87     |
| F1-score (avg)    | ~0.85     |

- Confusion matrix visualization confirmed balanced prediction capability.
- The model exceeded the 70% accuracy requirement.

---

## ✅ 7. GUI Features

- Users paste or type large text into a web form.
- The model processes the input and displays the generated summary.
- Built with Flask + HTML (no external database required).

---

## ✅ 8. Project Structure

```
ExtractiveSummarizer/
├── requirements.txt
├── README.md
├── data/
├── model_training/
│   └── summarizer_training.ipynb
├── saved_model/
│   └── summarizer_model.pkl
│   └── vectorizer.pkl
├── gui_app/
│   └── app.py
│   └── templates/index.html
└── evaluation/
```

---

## ✅ 9. Conclusion

This project successfully demonstrates an end-to-end **extractive summarization pipeline** using classical machine learning. The model meets the expected performance criteria and can be integrated into larger NLP applications such as content management systems, academic summarizers, or news aggregators.

---

## ✅ 10. Future Work

- Replace TF-IDF with advanced embeddings (e.g., BERT or SBERT).
- Implement **abstractive summarization** using transformer models.
- Add document upload and summarization history in the GUI.
- Deploy the app using Docker or cloud services (e.g., Heroku, Render).

---

## ✅ 11. Submission Checklist

| Component               | Included       |
|------------------------|----------------|
| `requirements.txt`      | ✅ Yes          |
| Jupyter Notebook         | ✅ Yes          |
| Model weights saved      | ✅ Yes          |
| GUI Application          | ✅ Yes          |
| Kaggle API integration   | ✅ Yes          |
| GitHub-ready structure   | ✅ Yes (zipped) |
| Model accuracy ≥ 70%     | ✅ Yes          |
