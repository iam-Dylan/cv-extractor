#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
from io import BytesIO
from pathlib import Path
import re
import regex
import streamlit as st
import io
from PIL import Image

# ---------- Optional deps ----------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from tika import parser as tika_parser
except Exception:
    tika_parser = None

# ---------- spaCy ----------
import spacy
from spacy.tokens import Doc

# ==========================
# Utilities
# ==========================
EMAIL_RE = re.compile(r"(?i)(?<![\w\.-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w\.-])")
NAME_TOKEN_RE = regex.compile(r"^\p{Lu}[\p{L}\p{M}’'´`-]{1,}$", flags=regex.UNICODE)

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n[ \t]*", "\n", s)
    return s.strip()

def extract_emails(text: str):
    emails = [m.group(1) for m in EMAIL_RE.finditer(text or "")]
    out, seen = [], set()
    for e in emails:
        e2 = e.strip().strip(".,;:()[]<>").lower()
        if e2 and e2 not in seen:
            out.append(e2); seen.add(e2)
    return out

def is_plausible_name_token(tok: str) -> bool:
    tok = tok.strip().strip(",.;:()[]{}<>|/\\")
    if len(tok) < 2: return False
    if tok.lower() in {"curriculum", "vitae", "resume", "cv", "profile"}: return False
    if len(tok) > 3 and tok.isupper(): return False
    return bool(NAME_TOKEN_RE.match(tok))

def score_name_candidate(candidate: str, idx: int) -> float:
    toks = [t for t in re.split(r"\s+", candidate.strip()) if t]
    if not (1 <= len(toks) <= 5): return -1e6
    plaus = sum(is_plausible_name_token(t) for t in toks)
    if plaus < max(1, len(toks) - 1): return -1e6
    return 1000.0 / (1 + idx) + min(len(toks), 3) * 2.0

def heuristic_name_from_text(text: str):
    if not text: return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    top = lines[:15]
    bad_kw = ("curriculum vitae", "resume", "profile", "objective")
    cands = []
    for i, line in enumerate(top):
        if any(k in line.lower() for k in bad_kw): continue
        line_wo_email = EMAIL_RE.sub(" ", line)
        tokens = [t for t in re.split(r"[ \t]", line_wo_email) if t]
        namey = [t for t in tokens if is_plausible_name_token(t)]
        if namey and len(namey) >= max(1, len(tokens) - 3):
            cands.append((line.strip(",.;- "), i))
    if not cands: return None
    best = max(cands, key=lambda x: score_name_candidate(x[0], x[1]))
    return best[0]

# ==========================
# PDF helpers
# ==========================
@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    # 1) Try PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            parts = []
            for page in doc:
                t = page.get_text("text") or ""
                if t: parts.append(t)
            doc.close()
            txt = "\n".join(parts)
            if txt.strip():
                return clean_text(txt)
        except Exception:
            pass
    # 2) Tika fallback (needs Java + tika server under the hood)
    if tika_parser is not None:
        try:
            parsed = tika_parser.from_buffer(pdf_bytes, service="text")
            return clean_text((parsed or {}).get("content") or "")
        except Exception:
            pass
    return ""

def render_pdf_preview(pdf_bytes: bytes, zoom: float = 2.0, max_pages: int = 10):
    """
    Trả về list[Image.Image] (PNG) render từ PDF.
    - zoom: phóng to (2.0 ~ 144 DPI x2 → rõ nét)
    - max_pages: giới hạn số trang để hiển thị (tránh nặng)
    """
    if fitz is None:
        return []

    imgs = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(zoom, zoom)
        total = min(len(doc), max_pages)
        for i in range(total):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("png"))
            img = Image.open(buf)
            imgs.append(img)
        doc.close()
    except Exception:
        pass
    return imgs

# ==========================
# Model loading
# ==========================
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str = "outputs/model"):
    # Try custom fine-tuned model first
    p = Path(model_dir)
    if p.exists():
        try:
            nlp = spacy.load(str(p))
            return nlp, "custom_model"
        except Exception:
            pass
    # Fallback multilingual
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
        return nlp, "xx_ent_wiki_sm"
    except Exception:
        return None, "none"

def name_from_doc(doc: Doc, prefer=("Name", "PERSON", "PER")):
    ents = {}
    for e in doc.ents:
        ents.setdefault(e.label_, []).append(e)
    for lab in prefer:
        if ents.get(lab):
            e = sorted(ents[lab], key=lambda x: x.start_char)[0]
            return e.text.strip()
    return None

def infer(text: str, nlp):
    emails = extract_emails(text)
    name = None
    if nlp:
        doc = nlp(text.replace("\n", " "))
        name = name_from_doc(doc)
    if not name:
        name = heuristic_name_from_text(text)
    return name, emails

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="CV Name & Email Extractor", page_icon="📄", layout="wide")
st.title("📄 CV Name & Email Extractor")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_dir = st.text_input("Model directory (optional)", "outputs/model")
    nlp, nlp_kind = load_model(model_dir)
    st.write(f"Model: `{nlp_kind}`")
    st.caption("Nếu không có model fine-tune ở outputs/model, app sẽ dùng `xx_ent_wiki_sm`.")

uploaded = st.file_uploader("Tải lên 1 file PDF", type=["pdf"])
if uploaded is not None:
    pdf_bytes = uploaded.read()

    # --- Preview PDF ---
    st.subheader("👀 Preview PDF")
    pages = render_pdf_preview(pdf_bytes, zoom=2.0, max_pages=10)

    if not pages:
        st.warning("Không render được ảnh xem trước (có thể thiếu PyMuPDF) — sẽ hiển thị kết quả trích xuất bên dưới.")
    else:
        # Cho người dùng chọn trang để xem (tránh dồn ảnh nặng)
        idx = 0
        if len(pages) > 1:
            idx = st.slider("Trang", 1, len(pages), 1) - 1
        st.image(pages[idx], caption=f"Page {idx+1}/{len(pages)}", width=None)

    # --- Extract ---
    with st.spinner("Đang trích xuất..."):
        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text:
            st.warning("Không đọc được text từ PDF (có thể là scan ảnh).")
        name, emails = infer(text, nlp)

    # --- Results ---
    st.subheader("🧾 Kết quả")
    col1, col2 = st.columns(2) 

    with col1:
        st.write("**Name**")
        st.write(name if name else "—")

    with col2:
        st.write("**Email(s)**")
        if emails:
            for e in emails:
                st.write(f"📧 {e}")
        else:
            st.write("—")

    # Optional download JSON
    out_json = {
        "file": uploaded.name,
        "engine": nlp_kind,
        "name": name,
        "emails": emails,
    }
    st.download_button(
        "⬇️ Tải JSON kết quả",
        data=bytes(str(out_json), "utf-8"),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
    )
else:
    st.info("Hãy upload một file PDF để bắt đầu.")
