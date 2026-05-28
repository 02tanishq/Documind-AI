# ✨ DocuMind AI

DocuMind AI is an AI-powered document understanding platform that performs OCR, document classification, and transformer-based summarization for PDFs and scanned documents.

The platform allows users to upload documents, extract text using OCR, classify document categories, generate AI summaries, and maintain user history through an interactive web dashboard.

---

# 🚀 Features

- 📄 OCR Text Extraction using Tesseract OCR
- 🤖 AI-powered Document Classification
- 📝 Transformer-based Text Summarization
- 🔐 User Authentication System
- 📚 Document History Tracking
- 🖼️ Multi-format File Support
- ⚡ Interactive Streamlit Interface
- 📑 PDF Processing Support
- 🧠 NLP + Machine Learning Pipeline

---

# 🧠 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| OCR Engine | Tesseract OCR |
| Classification | XGBoost |
| Feature Extraction | TF-IDF |
| Summarization | Hugging Face Transformers (DistilBART) |
| Database | SQLite |
| PDF Processing | pdf2image |
| ML Serialization | Joblib |
| Language | Python |

---

# 🏗️ System Architecture

```text
Document Upload
       ↓
 OCR Text Extraction
       ↓
 TF-IDF Vectorization
       ↓
XGBoost Classification
       ↓
Transformer Summarization
       ↓
 Result Dashboard
```

---

# 📂 Supported File Formats

- PDF
- PNG
- JPG / JPEG
- TIFF

---

# 📸 Application Features

## 🔐 Authentication System
- User Signup/Login
- Password Hashing
- Session Management

## 📄 Document Processing
- Upload scanned documents or PDFs
- Extract text using OCR
- AI-powered classification
- Automatic summarization

## 📚 User History
- Stores previously analyzed documents
- Displays processing history

---

# 📁 Project Structure

```text
Documind-AI/
│
├── app.py
├── requirements.txt
├── packages.txt
├── render.yaml
├── .streamlit/
│   └── config.toml
├── xgb_model_new.pkl
├── tfidf_vectorizer_new.pkl
├── label_encoder_new.pkl
├── OCR Text Extraction.ipynb
├── classification.ipynb
├── user_data.db
└── README.md
```

---

# 🧠 Machine Learning Pipeline

## OCR Extraction
Uses Tesseract OCR to convert scanned documents and PDFs into machine-readable text.

## Document Classification
- TF-IDF Vectorization
- XGBoost Classification Model

## AI Summarization
Uses Hugging Face DistilBART transformer model for abstractive summarization.

---

# 🔐 Authentication & Database

- SQLite-based authentication system
- Password hashing using SHA256
- Session-based login handling
- User-specific history tracking

---

# 💡 Use Cases

- Invoice Processing
- Legal Document Analysis
- Research Paper Summarization
- Academic OCR Systems
- AI-based PDF Understanding
- Automated Document Intelligence

---

# 📈 Future Improvements

- Named Entity Recognition (NER)
- Multi-language OCR
- RAG-based Document Q&A
- Cloud Database Integration
- Vector Database Support
- Export Summaries as PDF

---

# 🛠️ Deployment

The application is deployed using Render.

---

# 🌐 Live Website

[🔗 Open Live App](https://documind-ai-weoj.onrender.com/)

---

# 🎥 Live Demo Video

[▶️ Watch Demo Video](https://drive.google.com/file/d/1_0cu69UiNTRg5O8GEyeoVyHNfwWeFcIz/view?usp=sharing)

---

# 👨‍💻 Author

**Tanishq Gupta**

- GitHub: https://github.com/02tanishq

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.
