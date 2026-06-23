# 📰 Fake News Detection System

An end-to-end NLP + Deep Learning pipeline for detecting fake news, built on the **LIAR dataset** — a benchmark dataset of ~12,800 manually fact-checked political statements. The project progresses from classical ML baselines to deep learning and transformer-based models, and ships as a deployed, interactive Streamlit app.

🔗 **Live App:** [Add your Streamlit Cloud URL here]

---

## 📌 Overview

Fake news spreads fast and influences public opinion at scale. This project builds and compares four different approaches to automatically classify a news statement as **FAKE** or **REAL**, then deploys the best-performing models behind a single interactive interface — letting a user pick any model, see its prediction confidence, and inspect *which words* drove that prediction via LIME-based explainability.

---

## 🗂️ Dataset

**Source:** [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) — UC Santa Barbara

| Field | Description |
|---|---|
| `statement` | The political claim being fact-checked |
| `label` | Original 6-class truthfulness rating (binarized for this project) |
| `subject`, `speaker`, `job`, `state`, `party` | Metadata about the speaker |
| `context` | Venue/setting of the statement |
| `barely_true_ct` ... `pants_fire_ct` | Speaker's historical credibility counts |

For this project, the original 6-class labels (`true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`) were collapsed into a **binary classification task**: `FAKE` vs `REAL`.

---

## 🏗️ Project Pipeline

```
Week 1   Data Preparation     EDA, text cleaning, feature engineering, train/val/test splits
Week 2   Baseline Models      TF-IDF + Logistic Regression, TF-IDF + Linear SVM
Week 3   Deep Learning        BiLSTM + Self-Attention (PyTorch), Fine-tuned BERT
Week 4   Deployment           Streamlit app with LIME explainability
```

### Week 1 — Data Preparation
- Automated LIAR dataset download
- Exploratory visualizations: label distribution, word counts, top speakers/subjects, word clouds, top bigrams/trigrams
- Cleaning pipeline: lowercasing → URL stripping → regex cleaning → stopword removal → lemmatization
- Engineered features: caps ratio, exclamation/question mark counts
- Stratified train/val/test splits

### Week 2 — Baseline Models
- **TF-IDF + Logistic Regression** — fast, interpretable baseline
- **TF-IDF + Linear SVM** — stronger margin-based baseline

### Week 3 — Deep Learning Models
- **BiLSTM + Self-Attention** *(custom PyTorch implementation)*
  - 2-layer bidirectional LSTM with a Bahdanau-style additive attention head
  - Trained from scratch on Keras-tokenized sequences
- **Fine-tuned BERT** *(`bert-base-uncased`)*
  - HuggingFace `transformers`, `[CLS]`-token classification head
  - Fine-tuned end-to-end on the LIAR statements

### Week 4 — Deployment
A 3-tab Streamlit application:

| Tab | What it does |
|---|---|
| 🔍 **Detect** | Pick any of the 4 trained models, enter a statement, get a prediction with a probability gauge and a **LIME**-based word-importance chart showing which words pushed the prediction toward FAKE or REAL |
| 📊 **Model Insights** | Side-by-side performance comparison across all 4 models |
| ℹ️ **How It Works** | Architecture overview, project checklist, and quick-start instructions |

---

## 🧠 Model Architectures

**BiLSTM + Attention**
```
Embedding → BiLSTM (2 layers, bidirectional) → Self-Attention → Dropout → Linear → Sigmoid
```
The attention layer learns to weight each word's contribution to the final prediction, collapsing the sequence of per-word vectors into a single context vector before classification.

**BERT**
```
bert-base-uncased → [CLS] token representation → Dropout → Linear → Sigmoid
```

---

## 🔍 Explainability — LIME

Rather than relying solely on internal attention weights (which only work for the BiLSTM model), the app uses **LIME (Local Interpretable Model-agnostic Explanations)** for word-importance — a technique that works identically across all 4 models. LIME perturbs the input statement (removing/altering words) and observes how each model's prediction shifts, ranking words by how much they push the prediction toward FAKE or REAL.

---

## 📁 Project Structure

```
fake_news_app/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── models/
│   ├── bilstm_best.pt          # Trained BiLSTM + Attention weights
│   ├── bert_best.pt            # Fine-tuned BERT weights
│   ├── tokenizer.pkl           # Keras tokenizer (for BiLSTM)
│   ├── lr_model.pkl            # Logistic Regression model
│   ├── svm_model.pkl           # Linear SVM model
│   └── tfidf.pkl               # TF-IDF vectorizer
└── README.md
```

---

## ⚙️ Tech Stack

- **Language:** Python
- **Classical ML:** scikit-learn (TF-IDF, Logistic Regression, Linear SVM)
- **Deep Learning:** PyTorch (custom BiLSTM + Attention), HuggingFace Transformers (BERT)
- **Preprocessing:** Keras Tokenizer, NLTK/spaCy-style cleaning pipeline
- **Explainability:** LIME
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Versioning:** Git, Git LFS (for large model checkpoints)

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 📊 Results Summary

| Model | Approach | Notes |
|---|---|---|
| Logistic Regression | TF-IDF | Fast, interpretable baseline |
| Linear SVM | TF-IDF | Stronger margin-based baseline |
| BiLSTM + Attention | Custom PyTorch | Learns word importance via attention |
| BERT | Fine-tuned `bert-base-uncased` | Pretrained language understanding |

*Detailed metrics (accuracy, precision, recall, F1, confusion matrices) are available in the Model Insights tab of the live app.*

---

## 🔮 Future Improvements

- Dynamic loading of evaluation charts (confusion matrices, ROC curves) into the Model Insights tab
- Swap `bert-base-uncased` for `distilbert-base-uncased` to reduce deployment memory footprint
- Expose models via a FastAPI service for programmatic access
- Incorporate speaker metadata (party, job, credibility history) as additional model features

---

## 📄 License

This project is for educational purposes, built on the publicly available LIAR dataset.
