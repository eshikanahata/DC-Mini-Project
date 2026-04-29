
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# !pip install -q torch transformers sentence-transformers accelerate \
#   langchain langchain-community chromadb pysqlite3-binary \
#   anthropic pandas scikit-learn tqdm rank-bm25

import os
import re
import pandas as pd
from tqdm import tqdm

import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("no gpu, switch to T4")


# using anthropic, swap out if you want openai or huggingface
import anthropic
os.environ["ANTHROPIC_API_KEY"] = "YOUR_KEY_HERE"
client = anthropic.Anthropic()

def call_llm(prompt, max_tokens=100):
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()

# openai version:
# from openai import OpenAI
# client = OpenAI(api_key="sk-...")
# def call_llm(prompt, max_tokens=100):
#     r = client.chat.completions.create(model="gpt-4o-mini",
#         messages=[{"role":"user","content":prompt}], max_tokens=max_tokens)
#     return r.choices[0].message.content.strip()

print(call_llm("say hi in one word"))


# load books and strip gutenberg header/footer
def load_book(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

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


# updating paths if you're mounting from drive
book_paths = {
    "The Count of Monte Cristo": "The_Count_of_Monte_Cristo.txt",
    "In Search of the Castaways": "In_search_of_the_castaways.txt"
}

books = {}
for name, path in book_paths.items():
    books[name] = load_book(path)
    print(f"{name}: {len(books[name].split()):,} words")


# chunking everything and keeping track of which book each chunk belongs to
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []
chunk_meta = []

for name, text in books.items():
    chunks = splitter.split_text(text)
    for i, c in enumerate(chunks):
        all_chunks.append(c)
        chunk_meta.append({"book": name, "chunk_id": i})
    print(f"{name} -> {len(chunks)} chunks")

print(f"total: {len(all_chunks)} chunks")


# building chromadb vector store
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

vector_db = Chroma.from_texts(
    texts=all_chunks,
    embedding=embedder,
    metadatas=chunk_meta,
    persist_directory="./chroma_store"
)

print(f"indexed {vector_db._collection.count()} chunks")


# building bm25 index alongside
bm25 = BM25Okapi([c.lower().split() for c in all_chunks])
print("bm25 ready")


# loading train/test — update paths as needed
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# train has labels, test doesn't
train_df["label_int"] = train_df["label"].map({"consistent": 1, "contradict": 0})

print(f"train: {len(train_df)} rows | {tr
ain_df['label'].value_counts().to_dict()}")
print(f"test:  {len(test_df)} rows (no labels)")
print(f"characters: {sorted(train_df['char'].unique())}")


# retrieval functions

def get_vector_chunks(query, k=5, book=None):
    if book:
        results = vector_db.similarity_search(query, k=k, filter={"book": book})
    else:
        results = vector_db.similarity_search(query, k=k)
    return [d.page_content for d in results]


def get_bm25_chunks(query, k=10):
    scores = bm25.get_scores(query.lower().split())
    top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(all_chunks[i], s) for i, s in top]


def get_hybrid_chunks(query, k=3, book=None):
    bm25_res = get_bm25_chunks(query, k=15)

    if book:
        vec_res = vector_db.similarity_search_with_score(query, k=15, filter={"book": book})
    else:
        vec_res = vector_db.similarity_search_with_score(query, k=15)

    # reciprocal rank fusion
    rrf_scores = {}
    for rank, (chunk, _) in enumerate(bm25_res):
        if book:
            idx = all_chunks.index(chunk) if chunk in all_chunks else -1
            if idx >= 0 and chunk_meta[idx]["book"] != book:
                continue
        rrf_scores[chunk] = rrf_scores.get(chunk, 0) + 1.0 / (rank + 60)

    for rank, (doc, _) in enumerate(vec_res):
        ch = doc.page_content
        rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (rank + 60)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [ch for ch, _ in ranked[:k]]


# test hybrid
sample = get_hybrid_chunks("Faria treasure prison", k=3, book="The Count of Monte Cristo")
print(sample[0][:200])


# prompts

def standard_prompt(claim, chunks):
    ctx = "\n\n".join([f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks)])
    return f"""You are a story fact-checker. Read the passages and decide if the claim is consistent.

STORY PASSAGES:
{ctx}

CLAIM: "{claim}"

Answer ONLY with '1' for Consistent or '0' for Inconsistent.
Answer:"""


def cot_prompt(claim, chunks):
    ctx = "\n\n".join([f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks)])
    return f"""You are a story fact-checker. Think carefully before answering.

STORY PASSAGES:
{ctx}

CLAIM: "{claim}"

Step 1 - What do the passages say about the character or event in the claim?
Step 2 - What does the claim say?
Step 3 - Do they match, or is there a contradiction? Check names, dates, numbers carefully.
Step 4 - Final answer: write only '1' if consistent, '0' if inconsistent.

Step 1:"""


# parsing the verdict out of the llm response

def parse(response):
    text = response.strip().lower()
    for ch in text[:20]:
        if ch == "1":
            return 1
        if ch == "0":
            return 0
    if "inconsistent" in text or "contradict" in text:
        return 0
    if "consistent" in text:
        return 1
    return -1


def parse_cot(response):
    text = response.lower()
    if "step 4" in text:
        after = text.split("step 4")[-1]
        if "inconsistent" in after or "contradict" in after:
            return 0
        if "0" in after[:80]:
            return 0
        if "1" in after[:80] or "consistent" in after:
            return 1
    if "final verdict" in text:
        after = text.split("final verdict")[-1]
        if "0" in after[:50] or "inconsistent" in after:
            return 0
        if "1" in after[:50] or "consistent" in after:
            return 1
    for ch in reversed(text):
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    return parse(response)


# main pipeline function

def verify(claim, book=None, k=3, mode="standard", retrieval="vector", verbose=False):
    if retrieval == "hybrid":
        ctx = get_hybrid_chunks(claim, k=k, book=book)
    elif retrieval == "bm25":
        ctx = [c for c, _ in get_bm25_chunks(claim, k=k)]
    else:
        ctx = get_vector_chunks(claim, k=k, book=book)

    if mode == "cot":
        prompt = cot_prompt(claim, ctx)
        max_tok = 400
    else:
        prompt = standard_prompt(claim, ctx)
        max_tok = 50

    if verbose:
        print(f"claim: {claim[:80]}...")
        print(f"book: {book} | retrieval: {retrieval} | mode: {mode}")
        for i, c in enumerate(ctx):
            print(f"  chunk {i+1}: {c[:100]}...")

    response = call_llm(prompt, max_tokens=max_tok)
    verdict = parse_cot(response) if mode == "cot" else parse(response)

    if verbose:
        labels = {1: "consistent", 0: "inconsistent", -1: "undecided"}
        print(f"response: {response[:150]}")
        print(f"verdict: {labels.get(verdict)}")

    return {"claim": claim, "verdict": verdict, "response": response, "context": ctx}


# test it
r = verify(train_df.iloc[0]["content"], book=train_df.iloc[0]["book_name"], verbose=True)


# evaluation loop

def run_eval(df, mode="standard", retrieval="vector"):
    rows = []
    has_labels = "label_int" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = verify(row["content"], book=row["book_name"], k=3,
                        mode=mode, retrieval=retrieval)
        entry = {
            "id": row.get("id", ""),
            "book": row["book_name"],
            "char": row["char"],
            "claim": row["content"][:100],
            "predicted": result["verdict"],
            "response": result["response"][:200],
            "top_chunk": result["context"][0][:100] if result["context"] else ""
        }
        if has_labels:
            entry["ground_truth"] = row["label_int"]
            entry["correct"] = int(result["verdict"] == row["label_int"])
        rows.append(entry)

    return pd.DataFrame(rows)


def accuracy_report(results, label=""):
    total = len(results)
    decided = results[results["predicted"] != -1]
    correct = decided["correct"].sum()
    acc = correct / len(decided) * 100 if len(decided) > 0 else 0

    cons = decided[decided["ground_truth"] == 1]
    cont = decided[decided["ground_truth"] == 0]
    cons_acc = cons["correct"].mean() * 100 if len(cons) > 0 else 0
    cont_acc = cont["correct"].mean() * 100 if len(cont) > 0 else 0

    print(f"\n--- {label} ---")
    print(f"overall:     {acc:.1f}%  ({int(correct)}/{len(decided)})")
    print(f"consistent:  {cons_acc:.1f}%  ({int(cons['correct'].sum())}/{len(cons)})")
    print(f"contradict:  {cont_acc:.1f}%  ({int(cont['correct'].sum())}/{len(cont)})")
    print(f"undecided:   {total - len(decided)}")

    return {"acc": acc, "cons_acc": cons_acc, "cont_acc": cont_acc,
            "undecided": total - len(decided)}


# part 1: baseline
print("running baseline eval...")
baseline = run_eval(train_df, mode="standard", retrieval="vector")
b_metrics = accuracy_report(baseline, "standard + vector (baseline)")


# failure analysis
# if the character's name shows up in the retrieved chunk, it's a reasoning failure
# if it doesn't, we probably pulled the wrong context entirely (retrieval failure)

def check_failures(results):
    failures = results[results["correct"] == 0].copy()
    if len(failures) == 0:
        print("no failures!")
        return failures

    ftypes = []
    for _, row in failures.iterrows():
        char = row["char"].split("/")[0].strip().lower()
        if char in row["top_chunk"].lower():
            ftypes.append("reasoning")
        else:
            ftypes.append("retrieval")

    failures["failure_type"] = ftypes

    print(f"\n{len(failures)} failures:")
    print(f"  retrieval: {ftypes.count('retrieval')}")
    print(f"  reasoning: {ftypes.count('reasoning')}")
    print()

    for _, row in failures.iterrows():
        gt = "consistent" if row["ground_truth"] == 1 else "contradict"
        pred = {1: "consistent", 0: "contradict", -1: "undecided"}.get(row["predicted"])
        print(f"[{row['failure_type']}] {row['char']} | expected {gt}, got {pred}")
        print(f"  {row['claim']}...")
        print(f"  context: {row['top_chunk'][:80]}...")
        print()

    return failures


baseline_failures = check_failures(baseline)


# look at the retrieved chunks for a few failures to understand what went wrong
for _, fail in baseline_failures.head(3).iterrows():
    full_claim = train_df[train_df["id"] == fail["id"]]["content"].values[0]
    print(f"\nclaim: {full_claim[:120]}...")
    print(f"char: {fail['char']} | failure: {fail['failure_type']}")
    chunks = get_vector_chunks(full_claim, k=3, book=fail["book"])
    for i, c in enumerate(chunks):
        print(f"  chunk {i+1}: {c[:120]}...")


# part 2: chain of thought
print("\nrunning cot eval...")
cot_results = run_eval(train_df, mode="cot", retrieval="vector")
cot_metrics = accuracy_report(cot_results, "cot + vector")

print(f"\ncomparison:")
print(f"{'':30} {'standard':>10} {'cot':>10} {'delta':>10}")
print(f"{'overall':30} {b_metrics['acc']:>9.1f}% {cot_metrics['acc']:>9.1f}% {cot_metrics['acc']-b_metrics['acc']:>+9.1f}%")
print(f"{'consistent':30} {b_metrics['cons_acc']:>9.1f}% {cot_metrics['cons_acc']:>9.1f}% {cot_metrics['cons_acc']-b_metrics['cons_acc']:>+9.1f}%")
print(f"{'contradict':30} {b_metrics['cont_acc']:>9.1f}% {cot_metrics['cont_acc']:>9.1f}% {cot_metrics['cont_acc']-b_metrics['cont_acc']:>+9.1f}%")


# cases where cot fixed a failure
print("\ncases fixed by cot:")
fixed = 0
for i in range(min(len(train_df), len(baseline), len(cot_results))):
    if baseline.iloc[i]["correct"] == 0 and cot_results.iloc[i]["correct"] == 1:
        fixed += 1
        row = train_df.iloc[i]
        print(f"\n  {row['char']} ({row['label']})")
        print(f"  claim: {row['content'][:120]}...")
        print(f"  standard said: {baseline.iloc[i]['predicted']} (wrong)")
        print(f"  cot said: {cot_results.iloc[i]['predicted']} (correct)")
        print(f"  cot reasoning: {cot_results.iloc[i]['response'][:300]}...")

print(f"\ntotal fixed: {fixed}")


# cases where cot made things worse
print("\ncases where cot regressed:")
regressed = 0
for i in range(min(len(train_df), len(baseline), len(cot_results))):
    if baseline.iloc[i]["correct"] == 1 and cot_results.iloc[i]["correct"] == 0:
        regressed += 1
        row = train_df.iloc[i]
        print(f"  {row['char']}: {row['content'][:100]}...")

if regressed == 0:
    print("  none, cot matched or beat baseline everywhere")


# annotated example showing why cot helped
for i in range(min(len(train_df), len(baseline), len(cot_results))):
    if baseline.iloc[i]["correct"] == 0 and cot_results.iloc[i]["correct"] == 1:
        row = train_df.iloc[i]
        print(f"\nexample where cot helped:")
        print(f"claim: {row['content']}")
        print(f"char: {row['char']} | book: {row['book_name']} | label: {row['label']}")
        print(f"\nstandard response: {baseline.iloc[i]['response']}")
        print(f"standard verdict: {baseline.iloc[i]['predicted']} <- wrong")
        print(f"\ncot response: {cot_results.iloc[i]['response'][:500]}")
        print(f"cot verdict: {cot_results.iloc[i]['predicted']} <- correct")
        break


# part 3: hybrid search
print("\nrunning hybrid + cot eval...")
hybrid_results = run_eval(train_df, mode="cot", retrieval="hybrid")
h_metrics = accuracy_report(hybrid_results, "cot + hybrid (bm25 + vector + rrf)")

hybrid_failures = check_failures(hybrid_results)

# how many retrieval failures did hybrid fix vs baseline
b_ret = len(baseline_failures[baseline_failures["failure_type"] == "retrieval"]) if "failure_type" in baseline_failures.columns else 0
h_ret = len(hybrid_failures[hybrid_failures["failure_type"] == "retrieval"]) if len(hybrid_failures) > 0 and "failure_type" in hybrid_failures.columns else 0

print(f"\nretrieval failures - baseline: {b_ret} | hybrid: {h_ret} | fixed: {b_ret - h_ret}")

print(f"\nfull comparison:")
print(f"{'method':40} {'overall':>8} {'consist':>8} {'contrad':>8}")
print(f"{'standard + vector':40} {b_metrics['acc']:>7.1f}% {b_metrics['cons_acc']:>7.1f}% {b_metrics['cont_acc']:>7.1f}%")
print(f"{'cot + vector':40} {cot_metrics['acc']:>7.1f}% {cot_metrics['cons_acc']:>7.1f}% {cot_metrics['cont_acc']:>7.1f}%")
print(f"{'cot + hybrid':40} {h_metrics['acc']:>7.1f}% {h_metrics['cons_acc']:>7.1f}% {h_metrics['cont_acc']:>7.1f}%")


# part 4: agent loop
# the agent can call search_story() multiple times before giving a verdict
# useful for claims that need more than one search to verify

AGENT_SYS = """You are a story fact-checker agent. Your job is to verify whether a claim about a character is consistent with the novel.

You have one tool: search_story(query) which returns relevant passages.

To search: <tool_call>search_story: YOUR QUERY</tool_call>
To give your verdict: <verdict>1</verdict> for consistent, <verdict>0</verdict> for inconsistent.

Always write your reasoning before the verdict tag. If the first search doesn't give enough evidence, search again with a more specific query."""


def run_agent(claim, book=None, max_iters=3, verbose=False):
    messages = [{
        "role": "user",
        "content": f'Verify this claim from "{book}":\n\n"{claim}"\n\nStart by searching for relevant passages.'
    }]

    verdict = -1
    reasoning = ""
    trace = []
    context_used = []

    for it in range(1, max_iters + 1):
        full_prompt = AGENT_SYS + "\n\n" + "\n".join(
            [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
             for m in messages]
        ) + "\nAssistant:"

        response = call_llm(full_prompt, max_tokens=500)
        messages.append({"role": "assistant", "content": response})
        trace.append(f"[iter {it}] {response[:300]}")

        if verbose:
            print(f"\niter {it}: {response[:200]}...")

        vm = re.search(r"<verdict>([01])</verdict>", response)
        if vm:
            verdict = int(vm.group(1))
            reasoning = response
            break

        tm = re.search(r"<tool_call>search_story:\s*(.+?)</tool_call>", response, re.DOTALL)
        if tm:
            q = tm.group(1).strip()
            if verbose:
                print(f"  searching: {q}")
            chunks = get_hybrid_chunks(q, k=3, book=book)
            context_used.extend(chunks)
            tool_out = "\n\n".join([f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks)])
            trace.append(f"  search({q}): {tool_out[:200]}...")
            messages.append({
                "role": "user",
                "content": f"Search results:\n{tool_out}\n\nContinue your analysis."
            })
        else:
            verdict = parse_cot(response)
            reasoning = response
            break

    if verdict == -1:
        verdict = parse_cot(messages[-1]["content"])

    return {
        "claim": claim,
        "verdict": verdict,
        "reasoning": reasoning,
        "trace": "\n".join(trace),
        "iters": it,
        "context": context_used
    }


# demo on one consistent and one contradict claim
demo = [
    train_df[train_df["label"] == "consistent"].iloc[0],
    train_df[train_df["label"] == "contradict"].iloc[0],
]

for row in demo:
    print(f"\nclaim: {row['content'][:120]}...")
    print(f"expected: {row['label']}")
    r = run_agent(row["content"], book=row["book_name"], verbose=True)
    labels = {1: "consistent", 0: "inconsistent", -1: "undecided"}
    print(f"agent verdict: {labels.get(r['verdict'])} | iterations: {r['iters']}")


# full trace for a multi-hop example
multi_hop = train_df[train_df["label"] == "contradict"].iloc[2]
result = run_agent(multi_hop["content"], book=multi_hop["book_name"], max_iters=3)
print(f"\nmulti-hop trace:")
print(f"claim: {multi_hop['content'][:200]}...")
print(f"expected: {multi_hop['label']}")
print(result["trace"])
print(f"verdict: {result['verdict']} | iters used: {result['iters']}")


# uncomment to run the full agent eval (takes ~15-20 mins)
# agent_rows = []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     r = run_agent(row["content"], book=row["book_name"])
#     agent_rows.append({
#         "id": row["id"], "char": row["char"],
#         "ground_truth": row["label_int"],
#         "predicted": r["verdict"],
#         "correct": int(r["verdict"] == row["label_int"]),
#         "iters": r["iters"]
#     })
# agent_df = pd.DataFrame(agent_rows)
# accuracy_report(agent_df, "agent + hybrid + cot")


# generate predictions for test.csv using our best pipeline
print("\ngenerating test.csv predictions...")

preds = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    r = verify(row["content"], book=row["book_name"], k=3, mode="cot", retrieval="hybrid")
    preds.append({
        "id": row["id"],
        "book_name": row["book_name"],
        "char": row["char"],
        "content": row["content"],
        "predicted": "consistent" if r["verdict"] == 1 else "contradict",
        "reasoning": r["response"][:300]
    })

pred_df = pd.DataFrame(preds)
pred_df.to_csv("test_predictions.csv", index=False)
print(f"saved test_predictions.csv")
print(pred_df["predicted"].value_counts())
