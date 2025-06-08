# IEPC Chat App — Semantic Search Over 35+ Years of Electric Propulsion Research 

**IEPC Chat App** is a PDF question-answering and semantic search tool built on the near complete archive of the International Electric Propulsion Conference (IEPC) proceedings (1988–2024).

Designed for **scientific rigour**, **transparent context use**, and **conversational research exploration**, this app ensures all answers are grounded strictly in the original papers — no hallucinations, no speculative ChatGPT knowledge.

---

##  Core Features

 **Strictly Context-Only Answers**  
→ The app answers only from the retrieved content of the IEPC PDFs. It will not hallucinate or invent information outside of the documents.

 **Conversational Memory**  
→ Full chat history is passed into the model for multi-turn, follow-up, or comparative queries. Ideal for deep-dive research discussions.

 **Metadata-Aware Querying**  
→ Supports structured queries about:
- **Publication year**
- **Authors**
- **Affiliations**
- **Countries**

With real summarisation based on *full paper content*, not just titles.

 **Semantic Vector Search Fallback**  
→ Uses OpenAI embeddings + FAISS to provide intelligent fallback for general Q&A when structured metadata isn’t enough.

 **Transparent Retrieval**  
→ All retrieved context chunks are displayed in expandable sections so users can inspect the exact source content used to generate the answer.

---

##  Project Structure

| File | Purpose |
|------|---------|
| `header_extractor.py` | Heuristically extracts title, authors, affiliations, and countries from page 1 |
| `ingest.py`            | (Optional) Builds your own FAISS index and metadata from PDFs |
| `streamlit_app.py`     | Streamlit frontend with ChatGPT-style UI and context-aware Q&A |
| `vector_store/`        | Populated at runtime with downloaded index + metadata |

---

##  Quickstart

### 1. Clone the Repo

```bash
git clone https://github.com/karadagbu/iepc-chat-app
cd iepc-chat-app
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Configure API Access

Create a `.env` file in the root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the App (Prebuilt Index Auto-Downloads)

```bash
streamlit run streamlit_app.py
```

The app will download from **Dropbox** into `vector_store/`:

- `index.faiss` — precomputed vector embeddings  
- `meta.jsonl`  — extracted metadata (title, authors, affiliations, countries, and text chunks)

---

##  Sample Questions

- “What is an external discharge plasma thruster?”
- "Why anomalous Bohm diffusion is called anomalous?"
- “Who studied microwave electrothermal propulsion?”
- “When should Hall‐effect thrusters be used instead of ion thrusters?”
- “How are future research trends shaping the field of electric propulsion?”
---

##  (Optional) Build Your Own Index from PDFs

If you wish to ingest your own archive:

1. Update requirement.txt file.
2. Place your PDFs inside the `pdfs/` folder.  
3. Ensure `.env` contains your OpenAI API key.  
4. Run:

   ```bash
   python ingest.py
   ```

This will extract, OCR, translate, chunk, embed, and output:

- `vector_store/index.faiss`
- `vector_store/meta.jsonl`

---

##  Tech Stack

-  **OpenAI GPT-4o** for NLP  
-  **text-embedding-3-small** for embeddings  
-  **FAISS** for similarity indexing  
-  **Tesseract OCR** for scanned PDFs  
-  **Streamlit** for UI  
-  **pdfplumber** & `langdetect` for parsing  

---

##  About

Built by **Burak Karadag**  
[linkedin.com/in/karadagbu](https://linkedin.com/in/karadagbu)

---

##  License

MIT License