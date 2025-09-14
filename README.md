# CV Name & Email Extractor

This project provides two main tools:
1. **Streamlit web app (`app.py`)** for uploading a CV (PDF) and interactively extracting candidate name & email.
2. **Command-line script (`cv_name_email_extractor.py`)** for training a spaCy NER model from annotations and batch-processing multiple CVs.

---

## Environment Setup

### Requirements
- Python **3.9 – 3.11**
- Recommended: create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, here are the core dependencies:

```txt
streamlit
spacy>=3.0
regex
pillow
pymupdf   # for PDF parsing & preview (optional but recommended)
tika      # fallback PDF parser (requires Java)
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

- Open the link shown in the terminal (default: `http://localhost:8501`).
- Upload a CV in PDF format.
- The app will:
  - Render the first pages of the PDF (if `pymupdf` installed).
  - Extract candidate name & email using:
    - A **custom fine-tuned spaCy model** in `outputs/model` (if available), or
    - Fallback multilingual model `xx_ent_wiki_sm`.
  - Show results in the UI and let you **download JSON output**.

---

## Training & Batch Extraction via CLI

### Train a model & infer a folder
```bash
python cv_name_email_extractor.py \
    --ann data/annotations.json \
    --pdf_dir data/pdfs \
    --out outputs/results.jsonl \
    --model_out outputs/model
```

### Train a model & infer a single file
```bash
python cv_name_email_extractor.py \
    --ann data/annotations.json \
    --pdf data/pdfs/cv_01.pdf \
    --out outputs/results.jsonl \
    --model_out outputs/model
```

### Use an existing trained model (skip training)
```bash
python cv_name_email_extractor.py \
    --model_in outputs/model \
    --pdf_dir data/pdfs \
    --out outputs/results.jsonl
```

- Results are written as JSON Lines (`.jsonl`), one CV per line with extracted `name` and `emails`.

---

## Project Structure
```
├── app.py                       # Streamlit web app
├── cv_name_email_extractor.py   # Training & batch extraction script
├── data/
│   ├── annotations.json         # Example NER training data
│   └── pdfs/                    # Input CV PDFs
├── outputs/
│   ├── model/                   # Saved spaCy model
│   └── results.jsonl            # Extraction results
```

---

## 📝 Notes
- If PDFs are scans/images (no text layer), you’ll need OCR (e.g. Tesseract). Currently OCR is not integrated.
- `pymupdf` is recommended for both **text extraction** and **preview rendering**. If missing, the app falls back to Tika (text only, no preview).
- Training data must contain `PERSON` labels in supported formats (`annotations.json`).
