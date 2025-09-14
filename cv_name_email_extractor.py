#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a spaCy NER model from annotations.json (PERSON) and extract Name + Emails
from either a single PDF or all PDFs in a folder.

Usage:
  # Train + batch infer folder:
  python scripts/cv_name_email_extractor.py \
      --ann data/annotations.json \
      --pdf_dir data/pdfs \
      --out outputs/results.jsonl \
      --model_out outputs/model

  # Train + infer single file:
  python scripts/cv_name_email_extractor.py \
      --ann data/annotations.json \
      --pdf data/pdfs/cv_01.pdf \
      --out outputs/results.jsonl \
      --model_out outputs/model

  # Reuse an existing trained model (skip training):
  python scripts/cv_name_email_extractor.py \
      --model_in outputs/model \
      --pdf_dir data/pdfs \
      --out outputs/results.jsonl
"""

import argparse, json, os, re, random, sys
from pathlib import Path
from typing import List, Tuple, Set

# --- PDF text extraction ---
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from tika import parser as tika_parser
except Exception:
    tika_parser = None

# --- spaCy 3.x ---
import spacy
from spacy.training.example import Example
from spacy.util import minibatch

# --- Unicode-aware regex ---
import regex  # pip install regex

EMAIL_RE = re.compile(r"(?i)(?<![\w\.-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w\.-])")

_EDGE_CHARS = ' \t\r\n"\'.,;:()[]<>“”‘’«»…'

NAME_TOKEN_RE = regex.compile(r"^\p{Lu}[\p{L}\p{M}’'´`-]{1,}$", flags=regex.UNICODE)

def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\r?\n[ \t]*", "\n", s)
    return s.strip()

def extract_emails(text: str) -> List[str]:
    """Trích email, chuẩn hóa lowercase, loại trùng theo thứ tự xuất hiện."""
    if not text:
        return []
    seen = set()
    out: List[str] = []
    add = seen.add  # bind local cho nhanh
    for m in EMAIL_RE.finditer(text):
        e = m.group(1).strip(_EDGE_CHARS).lower()
        if e and e not in seen:
            add(e)
            out.append(e)
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

def extract_text_from_pdf(pdf_path: Path) -> str:
    # 1) PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
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
    # 2) Tika fallback
    if tika_parser is not None:
        try:
            parsed = tika_parser.from_file(str(pdf_path), service="text")
            return clean_text((parsed or {}).get("content") or "")
        except Exception:
            pass
    return ""

# --------- Load training data from annotations.json ----------
def load_training_data(ann_path: Path):
    """
    Hỗ trợ 3 định dạng:
      A) JSON array: [ [text, {"entities":[[s,e,label],...]}], ... ]
      B) JSONL: mỗi dòng {"content": "...", "annotation":[{"label":[...], "points":[{"start":..,"end":..,"text":..}]}]}
      C) JSON object: {"classes":[...], "annotations":[ [text, {"entities":[[s,e,label],...]}], ... ]}
    Trả về: (train_data, labels)
    """
    raw = ann_path.read_text(encoding="utf-8", errors="ignore").strip()
    train, labels = [], set()

    def push(text, ents_triplets):
        ents = []
        for s, e, lab in ents_triplets:
            if s is None or e is None: 
                continue
            if e <= s: 
                continue
            ents.append((int(s), int(e), str(lab)))
            labels.add(str(lab))
        if text and ents:
            train.append((text, {"entities": ents}))

    # --- Thử load full JSON ---
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None

    # Case C: object có 'annotations'
    if isinstance(obj, dict) and "annotations" in obj:
        anns = obj.get("annotations", [])
        for item in anns:
            # kỳ vọng item = [text, {"entities":[[s,e,label], ...]}]
            if isinstance(item, list) and len(item) == 2:
                text, ann = item
                ents = []
                for trip in ann.get("entities", []):
                    if isinstance(trip, (list, tuple)) and len(trip) == 3:
                        s, e, lab = trip
                        ents.append((s, e, lab))
                push(text, ents)
        return train, labels

    # Case A: mảng [ [text, {"entities":[...]}], ... ]
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, list) and len(item) == 2:
                text, ann = item
                ents = []
                for s, e, lab in ann.get("entities", []):
                    ents.append((s, e, lab))
                push(text, ents)
        return train, labels

    # Case B: JSONL
    for line in raw.splitlines():
        line = line.strip()
        if not line: 
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = row.get("content", "")
        ents = []
        for ann in row.get("annotation", []):
            for lab in ann.get("label", []):
                for p in ann.get("points", []):
                    s, e = p.get("start"), p.get("end")
                    if isinstance(s, int) and isinstance(e, int):
                        # thường 'end' inclusive -> +1 thành exclusive
                        ents.append((s, e + 1, lab))
        push(text, ents)

    return train, labels


# --------- Train spaCy NER (xx tokenizer, Unicode) ----------
def train_ner(train_data, labels, n_iter=20, seed=42, lang="xx"):
    random.seed(seed); spacy.util.fix_random_seed(seed)
    nlp = spacy.blank(lang)
    ner = nlp.add_pipe("ner")
    if not labels:
        labels = {"PERSON"}
    for lab in labels:
        ner.add_label(lab)

    with nlp.select_pipes(disable=[p for p in nlp.pipe_names if p != "ner"]):
        optimizer = nlp.initialize(lambda: (Example.from_dict(nlp.make_doc(t), a) for t, a in train_data))
        for epoch in range(1, n_iter + 1):
            losses = {}
            random.shuffle(train_data)
            for batch in minibatch(train_data, size=16):
                examples = [Example.from_dict(nlp.make_doc(t), a) for t, a in batch]
                nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)
            print(f"[train] epoch {epoch}/{n_iter} losses={losses}")
    return nlp

def name_from_doc(doc, prefer=("Name", "PERSON", "PER")):
    ents = {}
    for e in doc.ents:
        ents.setdefault(e.label_, []).append(e)
    for lab in prefer:
        if ents.get(lab):
            e = sorted(ents[lab], key=lambda x: x.start_char)[0]
            return e.text.strip()
    return None

def infer_text(text: str, nlp):
    emails = extract_emails(text)
    name = None
    if nlp:
        doc = nlp(text.replace("\n", " "))
        name = name_from_doc(doc)
    if not name:
        name = heuristic_name_from_text(text)
    return name, emails

def iter_pdfs(pdf_dir: Path):
    for root, _, files in os.walk(str(pdf_dir)):
        for f in files:
            if f.lower().endswith(".pdf"):
                yield Path(root) / f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", help="annotations.json (train data). Omit if --model_in used.")
    ap.add_argument("--model_out", default="outputs/model", help="Where to save trained model")
    ap.add_argument("--model_in", help="Use existing model dir instead of training")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pdf", help="Single PDF path")
    ap.add_argument("--pdf_dir", help="Folder containing PDFs")
    ap.add_argument("--out", default="outputs/results.jsonl", help="Output JSONL file")
    args = ap.parse_args()

    # Prepare output dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or train model
    if args.model_in:
        nlp = spacy.load(args.model_in)
        print(f"[info] Loaded model from {args.model_in}")
    else:
        if not args.ann:
            print("[ERR] --ann is required when not using --model_in", file=sys.stderr)
            sys.exit(1)
        ann_path = Path(args.ann)
        if not ann_path.exists():
            print(f"[ERR] annotations not found: {ann_path}", file=sys.stderr); sys.exit(1)
        train_data, labels = load_training_data(ann_path)
        if not train_data:
            print("[ERR] No training data parsed from annotations.json", file=sys.stderr); sys.exit(1)
        print(f"[info] Train samples: {len(train_data)} | labels={sorted(labels)}")
        nlp = train_ner(train_data, labels, n_iter=args.epochs, lang="xx")
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(str(model_dir))
        print(f"[info] Model saved to {model_dir.resolve()}")

    # Collect pdfs
    pdfs = []
    if args.pdf:
        p = Path(args.pdf); 
        if not p.exists(): 
            print(f"[ERR] PDF not found: {p}", file=sys.stderr); sys.exit(1)
        pdfs.append(p)
    if args.pdf_dir:
        d = Path(args.pdf_dir)
        if not d.exists():
            print(f"[ERR] pdf_dir not found: {d}", file=sys.stderr); sys.exit(1)
        pdfs.extend(list(iter_pdfs(d)))
    if not pdfs:
        print("[ERR] Provide --pdf or --pdf_dir", file=sys.stderr); sys.exit(1)

    # Inference & write JSONL
    with out_path.open("w", encoding="utf-8") as fw:
        for pdf in pdfs:
            text = extract_text_from_pdf(pdf)
            if not text:
                print(f"[warn] Empty text from {pdf}")
            name, emails = infer_text(text, nlp)
            rec = {"file": str(pdf), "name": name, "emails": emails}
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(rec)

    print(f"[ok] Wrote results to {out_path.resolve()}")

if __name__ == "__main__":
    main()
