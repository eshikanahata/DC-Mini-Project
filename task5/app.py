import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import re
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
tf_logging.set_verbosity_error()


# ── Config ────────────────────────────────────────────────────────────────────

VECTORDB_DIR = "./vectordb"


# ── Cached model loads ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Qwen3-1.7B model...")
def load_model():
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
    return tokenizer, model

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource(show_spinner="Loading cross-encoder...")
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── LLM ───────────────────────────────────────────────────────────────────────

def call_llm(prompt, max_new_tokens=200, thinking=False):
    tokenizer, model = load_model()
    messages = [
        {"role": "system", "content": "You are a story fact-checker. Follow the user's instruction exactly."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Vector DB ─────────────────────────────────────────────────────────────────

def build_story_db(story_name, raw_text):
    embedding_model = load_embedding_model()
    db_path = os.path.join(VECTORDB_DIR, story_name)
    os.makedirs(db_path, exist_ok=True)
    chunks = splitter.split_text(raw_text)
    with open(os.path.join(db_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=db_path)
    return len(chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _tokenize(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def retrieve_chunks(query, story_name, k=4, k_candidates=20, rerank_top_n=10):
    embedding_model = load_embedding_model()
    cross_encoder = load_cross_encoder()

    db_path = os.path.join(VECTORDB_DIR, story_name)
    vdb = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    with open(os.path.join(db_path, "chunks.json"), "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    vector_docs = vdb.similarity_search(query, k=min(k_candidates, len(all_chunks)))
    vector_ranked = [doc.page_content for doc in vector_docs]

    tokenized_chunks = [_tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_candidates]
    bm25_ranked = [all_chunks[i] for i in bm25_indices]

    # RRF fusion
    scores = {}
    for ranked in (vector_ranked, bm25_ranked):
        for rank, chunk in enumerate(ranked):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (rank + 60)
    fused = [chunk for chunk, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)][:k_candidates]

    # Cross-encoder rerank
    rerank_pool = fused[:min(rerank_top_n, len(fused))]
    if rerank_pool:
        pairs = [(query, chunk) for chunk in rerank_pool]
        ce_scores = cross_encoder.predict(pairs)
        reranked = sorted(zip(rerank_pool, ce_scores), key=lambda item: item[1], reverse=True)
        fused = [chunk for chunk, _ in reranked] + fused[len(rerank_pool):]

    seen = set()
    unique = []
    for chunk in fused:
        if chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
        if len(unique) == k:
            break
    return unique


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_standard_prompt(claim, context_chunks):
    context_str = "\n\n".join([f"[Passage {i+1}] : {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""You are a story fact-checker. Carefully read the passages below and decide if the claim is consistent with them.

    story passages : 
    {context_str}

    claim :
    "{claim}"

    is this claim consistent with the story passages? answer with only '1' for consistent or '0' for inconsistent.
    Don't answer with "no" or "yes". use only 1 and 0. output '-1' if unsure/undecided.
    Answer : """

    return prompt


def build_cot_prompt(claim, context_chunks):
    context_str = "\n\n".join([f"[Passage {i+1}] : {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""You are a story fact-checker. You must reason briefly then give a verdict.

STORY PASSAGES:
{context_str}

CLAIM TO VERIFY: "{claim}"

Follow these steps. Keep each step to ONE sentence only:

Step 1 — What do the passages say about this topic? (one sentence)
Step 2 — What does the claim say? (one sentence)
Step 3 — Do they match or contradict? (one sentence)
Step 4 — VERDICT: 1 (consistent) or VERDICT: 0 (inconsistent)

Step 1: 
Step 2: 
Step 3: 
VERDICT: 
"""

    return prompt


def build_agent_prompt(claim, context_chunks, attempt=1):
    context_str = "\n\n".join([f"[Passage {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    return f"""You are a story fact-checker.

Use the passages to decide whether the claim is supported.

STORY PASSAGES:
{context_str}

CLAIM: "{claim}"

Return exactly one token:
1 = supported / consistent
0 = contradicted / inconsistent
NEED_MORE_INFO = only if the passages are clearly insufficient

If you choose NEED_MORE_INFO, add a second line:
FOLLOW_UP_QUERY: <short search query>
"""


def build_agent_verdict_prompt(claim, context_chunks):
    context_str = "\n\n".join([f"[Passage {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
    return f"""You are a story fact-checker.

Use the passages to decide if the claim is supported.

STORY PASSAGES:
{context_str}

CLAIM: "{claim}"

Return exactly one token: 1 or 0.
Do not explain.
"""


# ── Parsers ───────────────────────────────────────────────────────────────────

_VERDICT_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+verdict|verdict|answer)\s*[:\-]\s*([01]|yes|no|true|false|supported|unsupported|consistent|inconsistent|contradict(?:s|ion)?)\s*$"
)
_STEP4_LINE_RE = re.compile(
    r"(?im)^\s*step\s*4\b.*?([01]|yes|no|true|false|supported|unsupported|consistent|inconsistent|contradict(?:s|ion)?)\s*$"
)
_STANDALONE_VERDICT_LINE_RE = re.compile(
    r"(?im)^\s*([01]|yes|no|true|false|supported|unsupported|consistent|inconsistent|contradict(?:s|ion)?)\s*$"
)
_BRACKETED_DIGIT_RE = re.compile(r"""(?x)
    (?:\[\s*['"]?([01]|yes|no|true|false)['"]?\s*\]) |
    (?:['"]([01]|yes|no|true|false)['"])
""")


def _normalize_token(token):
    token = token.strip().lower()
    if token in {"1", "yes", "true", "supported", "consistent"}:
        return 1
    if token in {"0", "no", "false", "unsupported", "inconsistent", "contradict", "contradiction", "contradicts"}:
        return 0
    return None


def _parse_binary_verdict(text, *, undecided=-1):
    if not isinstance(text, str) or not text.strip():
        return undecided

    matches = list(_VERDICT_LINE_RE.finditer(text))
    if matches:
        verdict = _normalize_token(matches[-1].group(1))
        if verdict is not None:
            return verdict

    matches = list(_STEP4_LINE_RE.finditer(text))
    if matches:
        verdict = _normalize_token(matches[-1].group(1))
        if verdict is not None:
            return verdict

    matches = list(_STANDALONE_VERDICT_LINE_RE.finditer(text))
    if matches:
        verdict = _normalize_token(matches[-1].group(1))
        if verdict is not None:
            return verdict

    matches = list(_BRACKETED_DIGIT_RE.finditer(text))
    if matches:
        for match in reversed(matches):
            token = next((group for group in match.groups() if group), None)
            verdict = _normalize_token(token) if token else None
            if verdict is not None:
                return verdict

    lower = text.lower()
    has_consistent = "consistent" in lower
    has_inconsistent = "inconsistent" in lower or "contradict" in lower or "contradiction" in lower
    if has_consistent and not has_inconsistent:
        return 1
    if has_inconsistent and not has_consistent:
        return 0

    if lower.startswith(("yes", "true", "supported")):
        return 1
    if lower.startswith(("no", "false", "unsupported")):
        return 0

    return undecided


def parse_verdict(text, *, undecided=-1):
    return _parse_binary_verdict(text, undecided=undecided)


def parse_verdict_cot(text, *, undecided=-1):
    return _parse_binary_verdict(text, undecided=undecided)


def parse_agent_response(response):
    import json as _json

    need_more_info = False
    follow_up_query = ""
    verdict = -1
    text = response.strip()
    lower = text.lower()

    if text.startswith("{") and text.endswith("}"):
        try:
            payload = _json.loads(text)
            need_more_info = bool(payload.get("need_more_info", False))
            follow_up_query = str(payload.get("follow_up_query", "") or "").strip()
            raw_verdict = payload.get("verdict", None)
            if raw_verdict in {0, 1}:
                verdict = int(raw_verdict)
            elif isinstance(raw_verdict, str) and raw_verdict.strip() in {"0", "1", "yes", "no", "true", "false"}:
                verdict = 1 if raw_verdict.strip().lower() in {"1", "yes", "true"} else 0
        except Exception:
            pass

    if verdict == -1:
        if "need_more_info" in lower or "need more info" in lower or "follow_up_query" in lower:
            need_more_info = True
        if "follow_up_query:" in lower:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("follow_up_query:"):
                    follow_up_query = stripped.split(":", 1)[-1].strip()

        verdict_tokens = {"1", "yes", "true", "supported", "consistent"}
        negative_tokens = {"0", "no", "false", "unsupported", "inconsistent", "contradict", "contradiction", "contradicts"}
        for line in reversed(text.splitlines()):
            token = line.strip().strip("\"'").lower()
            if token in verdict_tokens:
                verdict = 1
                break
            if token in negative_tokens:
                verdict = 0
                break

        if verdict == -1:
            if lower.startswith(tuple(verdict_tokens)):
                verdict = 1
            elif lower.startswith(tuple(negative_tokens)):
                verdict = 0

        if verdict == -1:
            if "consistent" in lower and "inconsistent" not in lower:
                verdict = 1
            elif "inconsistent" in lower or "contradict" in lower or "contradiction" in lower:
                verdict = 0

    return {
        "need_more_info": need_more_info,
        "follow_up_query": follow_up_query,
        "verdict": verdict,
    }


def build_follow_up_query(claim, context_chunks):
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "at", "from",
        "his", "her", "their", "he", "she", "it", "they", "was", "were", "is", "are", "be", "been",
        "had", "has", "have", "as", "this", "that", "these", "those", "after", "before", "when", "while",
        "who", "whom", "whose", "what", "which", "where", "why", "how", "into", "over", "under", "about",
        "up", "down", "out", "off", "than", "then", "there", "here", "also", "very", "more", "most",
    }
    tokens = []
    for source_text in [claim] + list(context_chunks):
        for token in re.findall(r"[A-Za-z0-9']+", str(source_text).lower()):
            if token not in stopwords and len(token) > 2:
                tokens.append(token)
    if not tokens:
        return claim
    return " ".join(tokens[:10])


# ── Agentic verification ─────────────────────────────────────────────────────

def verify_claim_agent(claim, story_name, max_rounds=2, status_writer=None):
    """Agentic loop: search, read, decide — or search again."""
    all_context = []
    query = claim
    response = ""
    followup_mode = False
    log = []

    for round_num in range(1, max_rounds + 1):
        if status_writer:
            status_writer(f"Round {round_num}: searching story...")

        new_chunks = retrieve_chunks(query, story_name)
        all_context.extend(new_chunks)

        seen = set()
        unique_context = []
        for chunk in all_context:
            if chunk not in seen:
                seen.add(chunk)
                unique_context.append(chunk)
        all_context = unique_context

        prompt = build_agent_verdict_prompt(claim, all_context) if followup_mode else build_agent_prompt(claim, all_context, attempt=round_num)

        if status_writer:
            status_writer(f"Round {round_num}: reasoning ({len(all_context)} passages)...")

        response = call_llm(prompt, max_new_tokens=96)
        parsed = parse_agent_response(response)

        log.append({
            "round": round_num,
            "query": query,
            "chunks_retrieved": len(new_chunks),
            "total_context": len(all_context),
            "need_more_info": parsed["need_more_info"],
            "follow_up_query": parsed["follow_up_query"],
            "verdict": parsed["verdict"],
            "response": response,
        })

        if parsed["need_more_info"]:
            query = parsed["follow_up_query"] or build_follow_up_query(claim, all_context)
            followup_mode = True
            if status_writer:
                status_writer(f"Round {round_num}: need more info, follow-up query: {query}")
            continue

        verdict = parsed["verdict"] if parsed["verdict"] in {0, 1} else parse_verdict(response)
        return {
            "claim": claim,
            "verdict": verdict,
            "raw_response": response,
            "context": all_context,
            "rounds": round_num,
            "log": log,
        }

    verdict = parse_verdict(response)
    return {
        "claim": claim,
        "verdict": verdict,
        "raw_response": response,
        "context": all_context,
        "rounds": max_rounds,
        "log": log,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Story Fact-Checker", layout="centered")
st.title("Story Consistency Checker")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Story")

    existing = []
    if os.path.isdir(VECTORDB_DIR):
        existing = sorted([d for d in os.listdir(VECTORDB_DIR) if os.path.isdir(os.path.join(VECTORDB_DIR, d))])

    source = st.radio("Source", ["Existing story", "Upload new"], label_visibility="collapsed")

    story_name = None

    if source == "Upload new":
        uploaded = st.file_uploader("Upload .txt file", type=["txt"])
        if uploaded:
            story_name = os.path.splitext(uploaded.name)[0]
            raw = uploaded.read().decode("utf-8")
            db_path = os.path.join(VECTORDB_DIR, story_name)
            if not os.path.isdir(db_path):
                with st.spinner("Indexing story..."):
                    n = build_story_db(story_name, raw)
                st.success(f"Indexed {n} chunks")
    else:
        if existing:
            story_name = st.selectbox("Select story", existing, label_visibility="collapsed")
        else:
            st.info("No stories indexed yet.")

    st.divider()
    prompt_mode = st.radio("Prompt", ["Agent", "Chain-of-Thought", "Standard"])

# ── Main ──────────────────────────────────────────────────────────────────────

with st.form("claim_form"):
    claim = st.text_input("Claim", placeholder="e.g. Elara grew up in an orphanage.")
    submitted = st.form_submit_button("Verify", disabled=(not story_name))

if submitted and claim and story_name:
    if prompt_mode == "Agent":
        with st.status("Verifying...", expanded=True) as status:
            def _write(msg):
                st.write(msg)
            result = verify_claim_agent(claim, story_name, status_writer=_write)
            st.write(f"Completed in {result['rounds']} round(s)")
            status.update(label="Done", state="complete")

        raw = result["raw_response"]
        verdict = result["verdict"]
        chunks = result["context"]
    else:
        ptype = "cot" if prompt_mode == "Chain-of-Thought" else "standard"

        with st.status("Verifying...", expanded=True) as status:
            st.write("Searching story...")
            chunks = retrieve_chunks(claim, story_name)
            st.write(f"Retrieved {len(chunks)} passages")

            if ptype == "cot":
                prompt = build_cot_prompt(claim, chunks)
                max_tok = 600
            else:
                prompt = build_standard_prompt(claim, chunks)
                max_tok = 20

            st.write("Reasoning...")
            raw = call_llm(prompt, max_new_tokens=max_tok, thinking=(ptype == "cot"))

            verdict = parse_verdict_cot(raw) if ptype == "cot" else parse_verdict(raw)
            status.update(label="Done", state="complete")

    # Verdict
    if verdict == 1:
        st.success("**Consistent**")
    elif verdict == 0:
        st.error("**Inconsistent**")
    else:
        st.warning("**Undecided**")

    # Reasoning
    with st.expander("Reasoning", expanded=True):
        if prompt_mode == "Agent":
            for step in result["log"]:
                st.markdown(f"---")
                st.markdown(f"**Round {step['round']}**")
                st.markdown(f"**Search query:** {step['query']}")
                st.markdown(f"**Chunks retrieved:** {step['chunks_retrieved']} (total context: {step['total_context']})")
                st.markdown(f"**LLM response:**")
                st.code(step["response"], language=None)
                if step["need_more_info"]:
                    st.info(f"Requested more info — follow-up query: *{step['follow_up_query']}*")
                else:
                    verdict_label = {1: "Consistent", 0: "Inconsistent", -1: "Undecided"}
                    st.markdown(f"**Parsed verdict:** {verdict_label.get(step['verdict'], 'Unknown')}")
        elif prompt_mode == "Chain-of-Thought":
            st.code(raw, language=None)
        else:
            st.write(raw)

    # Passages
    with st.expander("Retrieved Passages"):
        for i, c in enumerate(chunks):
            st.caption(f"Passage {i+1}")
            st.text(c)
