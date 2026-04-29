import streamlit as st
import os
import re
import hashlib
import torch
import pandas as pd

st.set_page_config(page_title="Story Consistency Checker", page_icon="📖", layout="wide")

st.markdown("""
<style>
.verdict { padding: 16px 22px; border-radius: 8px; font-size: 1.35rem;
           font-weight: 700; text-align: center; margin: 8px 0; }
.consistent   { background: #dcfce7; color: #166534; border: 2px solid #86efac; }
.inconsistent { background: #fee2e2; color: #991b1b; border: 2px solid #fca5a5; }
.undecided    { background: #fef9c3; color: #854d0e; border: 2px solid #fde047; }
.chunk { background: #f0f4ff; border-left: 4px solid #4f46e5;
         padding: 10px 14px; border-radius: 4px; font-size: 0.85rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_embedder():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )


def strip_gutenberg(text):
    for m in ["*** START OF THE PROJECT GUTENBERG EBOOK",
              "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
        if m in text:
            text = text[text.index(m):]
            text = text[text.index("\n")+1:]
            break
    for m in ["*** END OF THE PROJECT GUTENBERG EBOOK",
              "*** END OF THIS PROJECT GUTENBERG EBOOK",
              "End of the Project Gutenberg EBook"]:
        if m in text:
            text = text[:text.index(m)]
            break
    return text.strip()


@st.cache_resource(show_spinner=False)
def build_index(text, _h):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from rank_bm25 import BM25Okapi
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    vdb = Chroma.from_texts(texts=chunks, embedding=load_embedder())
    bm25 = BM25Okapi([c.lower().split() for c in chunks])
    return chunks, vdb, bm25


def call_llm(prompt, max_tokens=400):
    key = st.session_state.get("api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return "no api key set"
    import anthropic
    return anthropic.Anthropic(api_key=key).messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text.strip()


def hybrid_search(query, chunks, vdb, bm25, k=3):
    scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:15]
    vec_top = vdb.similarity_search_with_score(query, k=15)

    rrf = {}
    for rank, (idx, _) in enumerate(bm25_top):
        rrf[chunks[idx]] = rrf.get(chunks[idx], 0) + 1 / (rank + 60)
    for rank, (doc, _) in enumerate(vec_top):
        rrf[doc.page_content] = rrf.get(doc.page_content, 0) + 1 / (rank + 60)

    return [c for c, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]


def parse_verdict(resp):
    t = resp.lower()
    if "step 4" in t:
        a = t.split("step 4")[-1]
        if "inconsistent" in a or "contradict" in a:
            return 0
        if "0" in a[:60]:
            return 0
        if "1" in a[:60] or "consistent" in a:
            return 1
    for c in t[:30]:
        if c == "1":
            return 1
        if c == "0":
            return 0
    if "inconsistent" in t or "contradict" in t:
        return 0
    if "consistent" in t:
        return 1
    return -1


AGENT_SYS = """You are a story fact-checker agent. Your job is to verify whether a backstory claim is consistent with the novel.

You have one tool: search the story using <tool_call>search_story: YOUR QUERY</tool_call>
Give your verdict using <verdict>1</verdict> (consistent) or <verdict>0</verdict> (inconsistent).

Always explain your reasoning before the verdict tag. Search again with a more specific query if the first result isn't enough."""


def run_agent(claim, chunks, vdb, bm25, status, trace_box):
    messages = [{"role": "user", "content": AGENT_SYS.replace("YOUR QUERY", claim) + f'\n\nClaim to verify: "{claim}"\n\nStart by searching for relevant passages.'}]
    traces = []
    verdict = -1
    reasoning = ""

    for it in range(1, 4):
        status.info(f"agent iteration {it}/3...")

        prompt = "\n".join(
            [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
             for m in messages]
        ) + "\nAssistant:"

        resp = call_llm(prompt, 500)
        messages.append({"role": "assistant", "content": resp})
        traces.append(f"**[{it}]** {resp[:400]}")
        trace_box.markdown("\n\n---\n\n".join(traces))

        vm = re.search(r"<verdict>([01])</verdict>", resp)
        if vm:
            verdict = int(vm.group(1))
            reasoning = resp
            status.success(f"done in {it} iteration(s)")
            break

        tm = re.search(r"<tool_call>search_story:\s*(.+?)</tool_call>", resp, re.DOTALL)
        if tm:
            q = tm.group(1).strip()
            status.info(f'searching: "{q[:50]}"...')
            results = hybrid_search(q, chunks, vdb, bm25)
            tool_out = "\n\n".join([f"[Passage {i+1}]: {c}" for i, c in enumerate(results)])
            traces.append(f"**search:** `{q[:50]}`\n\n{tool_out[:300]}...")
            trace_box.markdown("\n\n---\n\n".join(traces))
            messages.append({"role": "user", "content": f"Search results:\n{tool_out}\n\nContinue."})
        else:
            verdict = parse_verdict(resp)
            reasoning = resp
            break

    if verdict == -1:
        verdict = parse_verdict(messages[-1]["content"])

    return verdict, reasoning


# layout
st.title("📖 Story Consistency Checker")
st.caption("Hybrid RAG + Chain-of-Thought + Agent Loop")

with st.sidebar:
    st.header("Settings")
    key = st.text_input("Anthropic API Key", type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", ""),
                        placeholder="sk-ant-...")
    if key:
        st.session_state["api_key"] = key
    st.divider()
    st.markdown("1. Upload a `.txt` story file\n2. Enter a claim\n3. Agent searches + reasons\n4. Verdict")
    st.divider()
    st.caption("MiniLM-L6-v2 · ChromaDB · BM25 · RRF · Claude Haiku")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Upload Story")
    files = st.file_uploader("Choose .txt file(s)", type=["txt"], accept_multiple_files=True)
    story = ""
    if files:
        for f in files:
            raw = f.read().decode("utf-8", errors="ignore")
            cleaned = strip_gutenberg(raw)
            story += "\n\n" + cleaned
            st.success(f"{f.name}: {len(cleaned):,} chars")
        with st.expander("Preview"):
            st.text(story[:500] + "...")

    st.subheader("Claim")
    claim_input = st.text_area(
        "Enter a backstory claim to verify:",
        height=110,
        placeholder='e.g. "Faria was arrested in 1815 and sent to the Château d\'If for life."'
    )
    go = st.button("Verify Claim", type="primary", use_container_width=True)

with right:
    st.subheader("Agent Activity")
    status_box = st.empty()
    trace_box = st.empty()
    st.subheader("Result")
    verdict_box = st.empty()
    reason_box = st.empty()

if go:
    if not story:
        st.error("upload a story file first")
    elif not claim_input.strip():
        st.error("enter a claim")
    elif not st.session_state.get("api_key"):
        st.error("add your api key in the sidebar")
    else:
        h = hashlib.md5(story[:1000].encode()).hexdigest()
        status_box.info("building index...")
        chunks, vdb, bm25 = build_index(story, h)
        status_box.info(f"{len(chunks)} chunks indexed, starting agent...")

        v, r = run_agent(claim_input.strip(), chunks, vdb, bm25, status_box, trace_box)

        css = {1: "consistent", 0: "inconsistent"}.get(v, "undecided")
        lbl = {1: "✅ CONSISTENT", 0: "❌ INCONSISTENT"}.get(v, "⚠️ UNDECIDED")
        verdict_box.markdown(f'<div class="verdict {css}">{lbl}</div>', unsafe_allow_html=True)

        if r:
            with reason_box.expander("Reasoning", expanded=True):
                st.text(r)

        with st.expander("Retrieved Passages"):
            for i, c in enumerate(hybrid_search(claim_input.strip(), chunks, vdb, bm25)):
                st.markdown(f'<div class="chunk"><b>Passage {i+1}</b><br>{c}</div>',
                            unsafe_allow_html=True)

st.divider()
with st.expander("Batch predictions from test.csv"):
    csv = st.file_uploader("Upload test.csv", type=["csv"], key="csv")
    if csv and story and st.session_state.get("api_key") and st.button("Run Batch"):
        df = pd.read_csv(csv)
        h = hashlib.md5(story[:1000].encode()).hexdigest()
        chunks, vdb, bm25 = build_index(story, h)
        rows = []
        prog = st.progress(0)
        for i, (_, row) in enumerate(df.iterrows()):
            ctx = hybrid_search(row["content"], chunks, vdb, bm25)
            ctx_str = "\n\n".join([f"[Passage {j+1}]: {c}" for j, c in enumerate(ctx)])
            prompt = f"Read the passages and decide if the claim is consistent.\n\nPASSAGES:\n{ctx_str}\n\nCLAIM: \"{row['content']}\"\n\nStep 1-4 reasoning, then verdict 1 or 0.\nStep 1:"
            resp = call_llm(prompt, 400)
            rows.append({
                "id": row.get("id", i),
                "char": row["char"],
                "content": row["content"][:80],
                "predicted": "consistent" if parse_verdict(resp) == 1 else "contradict"
            })
            prog.progress((i + 1) / len(df))
        out = pd.DataFrame(rows)
        st.dataframe(out)
        st.download_button("Download CSV", out.to_csv(index=False),
                           "test_predictions.csv", "text/csv")
