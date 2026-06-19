import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, pad_mask=None):
        scores = self.attn_fc(lstm_out).squeeze(-1)          # (B, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)              # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, 2H)
        return context, weights
    
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                 bidirectional=True, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        pad_mask        = (x != 0).float()
        emb             = self.dropout(self.embedding(x))
        out, _          = self.lstm(emb)
        context, attn_w = self.attention(out, pad_mask)
        logit           = self.fc(self.dropout(context))
        return logit.squeeze(-1), attn_w
    
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(768, 1)   # BERT hidden size is always 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token → (B, 768)
        return self.fc(self.dropout(cls)).squeeze(-1)
    
# ── Load Models (cached so they load only once) ───────────────────────────────

@st.cache_resource
def load_all_models():
    # Keras tokenizer
    with open("models/tokenizer.pkl", "rb") as f:
        keras_tok = pickle.load(f)
 
    # TF-IDF + LR + SVM
    with open("models/tfidf.pkl",     "rb") as f: tfidf     = pickle.load(f)
    with open("models/lr_model.pkl",  "rb") as f: lr_model  = pickle.load(f)
    with open("models/svm_model.pkl", "rb") as f: svm_model = pickle.load(f)
 
    # BiLSTM
    bilstm = BiLSTMAttention(
    vocab_size = 20001,
    embed_dim  = 128,
    hidden_dim = 128,
    num_layers = 2,
    dropout    = 0.4)
    ckpt   = torch.load("models/bilstm_best.pt", map_location="cpu")
    bilstm.load_state_dict(ckpt["model_state_dict"])
    bilstm.eval()
 
    # BERT tokenizer
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
 
    # BERT model
    bert = BertClassifier()
    bert_path = hf_hub_download(
    repo_id="GCharanteja/fake-news-bert",
    filename="bert_best.pt"
    )
    bert.load_state_dict(
    torch.load(
        bert_path,
        map_location="cpu"
        )    
    )

    bert.eval()
 
    return keras_tok, tfidf, lr_model, svm_model, bilstm, bert_tok, bert
 
keras_tok, tfidf, lr_model, svm_model, bilstm, bert_tok, bert = load_all_models()

# ── Prediction Functions ───────────────────────────────────────────────────────
def pad_sequences_custom(sequences, maxlen):
    padded = []

    for seq in sequences:

        seq = seq[:maxlen]

        seq = seq + [0] * (maxlen - len(seq))

        padded.append(seq)

    return np.array(padded)




def predict_lr_svm(text, model, is_svm=False):
    vec = tfidf.transform([text])
    
    if is_svm:
        score = model.decision_function(vec)[0]
        prob  = 1 / (1 + np.exp(-score))   # sigmoid to convert to 0-1 range
    else:
        prob = model.predict_proba(vec)[0][1]
    
    return prob
 
def predict_bilstm(text):
    seq     = keras_tok.texts_to_sequences([text])
    padded = pad_sequences_custom(
    seq,
    maxlen=50
    )
    x       = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        logit, attn_w = bilstm(x)
    prob    = torch.sigmoid(logit).item()
    weights = attn_w.squeeze(0).numpy()         # (50,)
    tokens  = keras_tok.sequences_to_texts(seq)[0].split()
    return prob, tokens, weights[:len(tokens)]
 
def predict_bert(text):
    enc = bert_tok(text, max_length=50, padding="max_length",
                   truncation=True, return_tensors="pt")
    with torch.no_grad():
        logit = bert(enc["input_ids"], enc["attention_mask"])
    prob = torch.sigmoid(logit).item()
    return prob


# ── Gauge Bar ─────────────────────────────────────────────────────────────────
def show_gauge(prob):
    label = "REAL ✅" if prob >= 0.5 else "FAKE 🚨"
    color = "#2ecc71"  if prob >= 0.5 else "#e74c3c"
    st.markdown(f"### Prediction: **{label}**")
    st.progress(prob)
    col1, col2, col3 = st.columns(3)
    col1.metric("FAKE probability", f"{(1-prob)*100:.1f}%")
    col2.metric("REAL probability", f"{prob*100:.1f}%")
    col3.metric("Confidence",       f"{max(prob, 1-prob)*100:.1f}%")
 
# ── Word Importance Chart ──────────────────────────────────────────────────────
def show_word_importance(tokens, weights):
    if len(tokens) == 0:
        st.info("No tokens to display.")
        return
 
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
 
    # Sort by importance
    sorted_idx     = np.argsort(weights)[::-1][:15]   # top 15 words
    sorted_tokens  = [tokens[i] for i in sorted_idx]
    sorted_weights = weights[sorted_idx]
 
    colors = ["#e74c3c" if w > weights.mean() else "#3498db" for w in sorted_weights]
 
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sorted_tokens[::-1], sorted_weights[::-1], color=colors[::-1])
    ax.set_xlabel("Attention Weight")
    ax.set_title("Word Importance (Attention Weights)")
    red_patch  = mpatches.Patch(color="#e74c3c", label="High importance")
    blue_patch = mpatches.Patch(color="#3498db", label="Low importance")
    ax.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    st.pyplot(fig)
 
# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Model Insights", "ℹ️ How It Works"])
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DETECT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("📰 Fake News Detector")
    st.markdown("Enter a news statement below and select a model to classify it.")
 
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "SVM", "BiLSTM + Attention", "BERT"]
    )
 
    user_input = st.text_area("Enter news statement here:", height=120,
                               placeholder="e.g. The president signed a new bill today...")
 
    if st.button("Detect", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a statement.")
        else:
            with st.spinner("Analyzing..."):
 
                if model_choice == "Logistic Regression":
                    prob = predict_lr_svm(user_input, lr_model)
                    show_gauge(prob)
                    st.info("ℹ️ Word importance chart is only available for BiLSTM model.")
 
                elif model_choice == "SVM":
                    prob = predict_lr_svm(user_input, svm_model, is_svm=True)   # add is_svm=True
                    show_gauge(prob)
                    st.info("ℹ️ Word importance chart is only available for BiLSTM model.")
 
                elif model_choice == "BiLSTM + Attention":
                    prob, tokens, weights = predict_bilstm(user_input)
                    show_gauge(prob)
                    st.markdown("### 🔠 Word Importance")
                    show_word_importance(tokens, weights)
 
                elif model_choice == "BERT":
                    prob = predict_bert(user_input)
                    show_gauge(prob)
                    st.info("ℹ️ Word importance chart is only available for BiLSTM model.")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("📊 Model Insights")
    st.markdown("Performance comparison across all models.")
 
    # Summary Table
    st.markdown("### Model Comparison")
    st.table({
        "Model"    : ["Logistic Regression", "SVM", "BiLSTM + Attention", "BERT"],
        "Accuracy" : ["~%", "~%", "60%", "~%"],
        "Macro F1" : ["~%", "~%", "59%", "~%"],
        "Notes"    : [
            "Fast, interpretable baseline",
            "Strong margin-based baseline",
            "Custom PyTorch with attention",
            "Pretrained transformer"
        ]
    })
    st.caption("Fill in your Week 2 accuracies in the table above.")
 
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("ℹ️ How It Works")
 
    st.markdown("### 🏗️ Architecture Overview")
    st.markdown("""
    | Model | How it works |
    |---|---|
    | **Logistic Regression** | TF-IDF converts text to word frequency vectors → LR classifies |
    | **SVM** | TF-IDF vectors → finds best decision boundary between FAKE and REAL |
    | **BiLSTM + Attention** | Embeds words → BiLSTM reads context → Attention picks important words → classifies |
    | **BERT** | Pretrained on 3.3B words → fine-tuned on your data → [CLS] token used for classification |
    """)
 
    st.markdown("### ✅ Project Checklist")
    st.markdown("""
    - [x] Week 1 — EDA + Data Preprocessing
    - [x] Week 2 — TF-IDF + Logistic Regression + SVM
    - [x] Week 3 — BiLSTM + Attention + BERT
    - [x] Week 4 — Streamlit Deployment
    """)
 
    st.markdown("### 🚀 Quick Start")
    st.code("""
# Install dependencies
pip install -r requirements.txt
 
# Run the app
streamlit run app.py
    """, language="bash")
