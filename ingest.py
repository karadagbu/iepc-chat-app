#!/usr/bin/env python3
"""
ingest.py (with resume, HNSW index, batch embeddings, chunk overlap, structured logging)
"""

import os
import glob
import json
import pathlib
import multiprocessing
import logging
import re

import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException
from openai import OpenAI, APIError, APIStatusError
import tiktoken
import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import pytesseract
from langdetect import detect, LangDetectException

from header_extractor import extract_header_heuristic as extract_header

# ─── CONFIG ─────────────────────────────────────────────
PDF_DIR           = "pdfs"
STORE_DIR         = "vector_store"
EMBED_MODEL       = "text-embedding-3-small"
CHUNK_TOKENS      = 512
CHUNK_OVERLAP     = 50
MIN_TRANSLATE_CHARS = 50
CHECKPOINT_FREQ   = 100
EMBED_BATCH_SIZE  = 50

LANG_CODE_TO_TESS = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "ru": "rus", "zh-cn": "chi_sim",
    "zh": "chi_sim", "ja": "jpn", "ko": "kor"
}
# ─── END CONFIG ─────────────────────────────────────────

# Set up structured logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("ingest")

load_dotenv()

INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH  = os.path.join(STORE_DIR, "meta.jsonl")
os.makedirs(STORE_DIR, exist_ok=True)

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def translate_to_english(text: str, client: OpenAI) -> str:
    messages = [
        {"role": "system", "content": "Translate to English exactly."},
        {"role": "user",   "content": text}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return text

def ocr_page_if_needed(page) -> str:
    try:
        pil = page.to_image(resolution=300).original
        text = pytesseract.image_to_string(pil, lang="eng").strip()
    except Exception as e:
        logger.warning("OCR initial (eng) failed: %s", e)
        return ""
    if len(text) < MIN_TRANSLATE_CHARS:
        return text
    lang = detect_language(text)
    tess = LANG_CODE_TO_TESS.get(lang)
    if tess and tess != "eng":
        try:
            alt = pytesseract.image_to_string(pil, lang=tess).strip()
            return alt or text
        except Exception:
            pass
    return text

def chunk_text(text: str, encoder: tiktoken.Encoding) -> list[str]:
    toks = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + CHUNK_TOKENS, len(toks))
        chunk = encoder.decode(toks[start:end])
        chunks.append(chunk)
        start += CHUNK_TOKENS - CHUNK_OVERLAP
    return chunks

def embed_chunks(chunks: list[str], client: OpenAI) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend(
                np.array(item.embedding, dtype="float32")
                for item in resp.data
            )
        except (APIError, APIStatusError) as e:
            logger.warning("Embedding batch failed: %s", e)
            vectors.extend(
                np.zeros((len(batch), 1536), dtype="float32")
            )
    return vectors

def get_year_from_filename(fn: str) -> str:
    m = re.search(r"IEPC[-_](\d{4})-", fn, re.IGNORECASE)
    return m.group(1) if m else None

def process_pdf(pdf_path: str):
    # Worker-local setup
    load_dotenv()
    client_w = OpenAI()
    encoder_w = tiktoken.encoding_for_model(EMBED_MODEL)

    filename = pathlib.Path(pdf_path).name
    year = get_year_from_filename(filename)
    hdr = extract_header(pdf_path)
    all_chunks: list[str] = []
    all_meta: list[dict]  = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg_num, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text() or ""
                except PdfminerException:
                    raw = ""
                text = raw.strip() or ocr_page_if_needed(page)
                if not text:
                    continue
                if len(text) >= MIN_TRANSLATE_CHARS and detect_language(text) != "en":
                    text = translate_to_english(text, client_w)
                for chunk in chunk_text(text, encoder_w):
                    meta = {
                        "source":       filename,
                        "year":         year,
                        "page":         pg_num,
                        "title":        hdr.get("title"),
                        "authors":      hdr.get("authors", []),
                        "affiliations": hdr.get("affiliations", []),
                        "countries":    hdr.get("countries", [])
                    }
                    all_chunks.append(chunk)
                    all_meta.append(meta)
    except Exception as e:
        logger.error("Failed PDF %s: %s", filename, e)
        return [], [], []

    vecs = embed_chunks(all_chunks, client_w)
    return vecs, all_chunks, all_meta

def main():
    # 1) Load or init HNSW index
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        all_meta = []
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                all_meta.append(json.loads(line))
        processed = {m["source"] for m in all_meta}
        logger.info("Resuming, %d PDFs already ingested", len(processed))
    else:
        index = faiss.IndexHNSWFlat(1536, 32)  # efConstruction=32
        all_meta = []
        processed = set()
        logger.info("Starting fresh ingestion")

    # 2) Discover new PDFs
    pdfs = glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
    to_proc = [p for p in pdfs if pathlib.Path(p).name not in processed]
    logger.info("Found %d total, %d new PDFs", len(pdfs), len(to_proc))

    meta_file = open(META_PATH, "a", encoding="utf-8")
    count = 0

    # 3) Parallel ingestion
    with multiprocessing.Pool() as pool:
        for vecs, chunks, metas in tqdm(
            pool.imap_unordered(process_pdf, to_proc),
            total=len(to_proc),
            desc="Ingesting"
        ):
            if not vecs:
                continue
            index.add(np.vstack(vecs))
            for txt, m in zip(chunks, metas):
                m["text"] = txt
                meta_file.write(json.dumps(m, ensure_ascii=False) + "\n")
            meta_file.flush()

            processed.add(metas[0]["source"])
            count += 1
            if count % CHECKPOINT_FREQ == 0:
                faiss.write_index(index, INDEX_PATH)
                logger.info("Checkpoint: %d PDFs", count)

    # 4) Final checkpoint
    faiss.write_index(index, INDEX_PATH)
    meta_file.close()
    logger.info("Finished %d PDFs. Index → %s, Metadata → %s",
                len(processed), INDEX_PATH, META_PATH)

if __name__ == "__main__":
    main()
