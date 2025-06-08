import os
import requests 
import urllib.request
import gdown
import re
import json
import numpy as np
import faiss
import gdown
import tiktoken
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIStatusError

# ─── STREAMLIT CONFIG ─────────────────────────────────
st.set_page_config(page_title="PDF Chat", page_icon="🚀")
st.title("🚀💬 Chat Over IEPC Archive")
st.subheader("by Burak Karadag")
# add one blank line
st.markdown("")  
#—or via HTML—
# st.markdown("<br>", unsafe_allow_html=True)

# ─── DOWNLOAD CONFIG ────────────────────────────────────
INDEX_URL = (
    "https://www.dropbox.com/scl/fi/spv06uacbqp4xtrf76ent/index.faiss"
    "?rlkey=hjcn2fipcjrllpsdxb8nzpfa4&st=rl9oj9mf&dl=1"
)
META_URL = (
    "https://www.dropbox.com/scl/fi/xuca73i20ay8nh29voa1o/meta.jsonl"
    "?rlkey=dx2yqlhhue2gf9aa7amt0tz0q&st=d41r6ic5&dl=1"
)
INDEX_PATH = "vector_store/index.faiss"
META_PATH  = "vector_store/meta.jsonl"

def download_http(path, url, min_size_mb=1):
    if not os.path.exists(path) or os.path.getsize(path) < min_size_mb * 1024 * 1024:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with st.spinner(f"Downloading {os.path.basename(path)}…"):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        if os.path.getsize(path) < min_size_mb * 1024 * 1024:
            st.error(f"Downloaded {path} is too small ({os.path.getsize(path)} bytes).")
            st.stop()

# Download vector store files if missing or too small
download_http(INDEX_PATH, INDEX_URL)
download_http(META_PATH, META_URL)

# ─── LOAD FAISS INDEX & METADATA ───────────────────────
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    st.error("❌ FAISS index failed to load (missing or corrupted).")
    st.exception(e)
    st.stop()

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
except Exception as e:
    st.error("❌ Metadata file failed to load.")
    st.exception(e)
    st.stop()


# ─── LOAD ENV & OPENAI CLIENT ─────────────────────────
load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
ENCODER     = tiktoken.encoding_for_model(EMBED_MODEL)



# ─── SESSION STATE ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_term" not in st.session_state:
    st.session_state.last_term = None

#k = st.sidebar.slider("Top-K snippets", 1, 10, 5)
k = 10


# ─── DISPLAY CHAT HISTORY ─────────────────────────────
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ─── USER INPUT ───────────────────────────────────────
user_input = st.chat_input("Ask a question about your PDFs…")
if not user_input:
    st.stop()

# echo user
st.session_state.history.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

# ─── TERM DETECTION ───────────────────────────────────
m = re.match(r"\s*what is (.+?)(\?|$)", user_input.strip().lower())
if m:
    st.session_state.last_term = m.group(1).strip()

# ─── PRONOUN RESOLUTION ───────────────────────────────
if st.session_state.last_term:
    user_input = re.sub(
        r"\b(it|this|that|they|their|its)\b",
        st.session_state.last_term,
        user_input,
        flags=re.IGNORECASE
    )

# ─── HANDLE TERM-ONLY QUERIES ─────────────────────────
if not re.search(r"\b(what|who|how|why|where|when|which|list|show|count|plot|trend)\b", user_input.lower()):
    user_input = f"What is {user_input.strip().rstrip('?')}?"

# ─── HANDLE COMMONALITY COMPARISON ────────────────────
freq_match = re.match(
    r"which (?:one )?(?:is )?(?:more (?:often used|common))\s+(.+?)\s+or\s+(.+?)\??$",
    user_input.strip().lower()
)
if freq_match:
    term1, term2 = freq_match.group(1), freq_match.group(2)
    cnt1 = sum(md.get("text", "").lower().count(term1) for md in metadata)
    cnt2 = sum(md.get("text", "").lower().count(term2) for md in metadata)
    more = term1 if cnt1 > cnt2 else term2
    reply = f"'{more}' is more common: {cnt1} vs {cnt2}."
    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.stop()

# ─── EMBED & SEARCH ───────────────────────────────────
try:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[user_input])
    qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    _, idxs = index.search(qvec, k)

    # ─── BUILD CONTEXT ─────────────────────────────────
    snippets, refs = [], []
    for i in idxs[0]:
        md = metadata[i]
        text = md.get("text", "")
        source = md.get("source", "Unknown")
        page = md.get("page", "?")
        snippet = text + f"\n\n— *{source}, p.{page}*"
        snippets.append(snippet)
        refs.append((source, page))
    context = "\n\n---\n\n".join(snippets)

    # ─── BUILD MESSAGES ────────────────────────────────
    messages = [{"role": "system", "content": "You are a helpful assistant. Use only the CONTEXT to answer."}]
    for turn in st.session_state.history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_input}"})

    # ─── CALL LLM ──────────────────────────────────────
    with st.chat_message("assistant"):
        acc = ""
        placeholder = st.empty()
        for chunk in client.chat.completions.create(model="gpt-4o", messages=messages, stream=True):
            delta = chunk.choices[0].delta.content
            if delta:
                acc += delta
                placeholder.markdown(acc)

        if refs:
            ref_block = "\n".join(f"- {s}, p.{p}" for s, p in dict.fromkeys(refs))
            placeholder.markdown(acc + "\n\n**References used:**\n" + ref_block)
            acc += "\n\n**References used:**\n" + ref_block

        st.session_state.history.append({"role": "assistant", "content": acc})

except (APIError, APIStatusError) as e:
    err = f"Error: {e}"
    st.session_state.history.append({"role": "assistant", "content": err})
    with st.chat_message("assistant"):
        st.markdown(err)
