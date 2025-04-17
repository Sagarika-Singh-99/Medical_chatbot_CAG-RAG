# Check paths carefully

# 1. CHECK DATASETS

from datasets import load_dataset
from datasets import load_from_disk

dataset = load_dataset("json", data_files="cp_datasets/medsquad_all/medquad_corpus.json")
# Column names: ['question', 'answer']

dataset = load_from_disk("cp_datasets/medmcqa_processed")
# Column names: ['context', 'question', 'answer']

dataset = load_from_disk("cp_datasets/bioasq_taskB_ALL")
# Column names: ['question', 'answer', 'type', 'documents', 'snippets']

import pandas as pd
df = pd.read_csv("cp_datasets/Kaggle_dataset_combined.csv")
# use: Dataset/symp_pre.csv
# Column names: ['Label', 'Def', 'Title', 'Source', 'Category']

# 2. ADD DATASETS INTO MEDICAL CORPUS

# Expected columns of the medical corpus - Column Names: ['doc_id', 'text', 'title', 'source', 'category', 'meta_json']

import json
import pandas as pd
import uuid
from tqdm import tqdm
from datasets import load_from_disk

# part 1
# Path to MedQuAD dataset
medquad_path = "cp_datasets/medsquad_all/medquad_corpus.json"

# Load data
with open(medquad_path, "r") as f:
    raw_data = json.load(f)

# Construct corpus entries
corpus_rows = []
doc_id_counter = 0

for item in tqdm(raw_data):
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    
    if question and answer:
        combined_text = f"Q: {question}\nA: {answer}"
        title = item.get("title", "unknown").strip()
        entry = {
            "doc_id": str(doc_id_counter),
            "text": combined_text, #we have combined question and answer into "text"
            "title": "medical_entity",
            "source": "medquad",
            "category": "faq",
            "meta_json": json.dumps(item)
        }
        corpus_rows.append(entry)
        doc_id_counter += 1

# Create DataFrame
corpus_df = pd.DataFrame(corpus_rows)

# Drop exact duplicate text entries
corpus_df.drop_duplicates(subset=["text"], inplace=True)

# Reset doc_ids after removing duplicates
corpus_df.reset_index(drop=True, inplace=True)
corpus_df["doc_id"] = corpus_df.index.astype(str)

# Save
corpus_df.to_pickle("final_corpus.pkl")
print("✅ Saved to final_corpus.pkl")

# part 2
# Load existing corpus
corpus_df = pd.read_pickle("final_corpus.pkl")
start_id = corpus_df.shape[0]

# Load medmcqa dataset from disk
medmcqa_ds = load_from_disk("cp_datasets/medmcqa_processed")
medmcqa_df = medmcqa_ds.to_pandas()  # No "train" split

# Build corpus rows
new_rows = []
for idx, row in medmcqa_df.iterrows():
    context = row.get("context", "").strip()
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()
    
    if question and answer:
        combined_text = f"Context: {context}\nQ: {question}\nA: {answer}"
        entry = {
            "doc_id": str(start_id + idx),
            "text": combined_text,
            "title": question[:50],
            "source": "medmcqa",
            "category": "mcqa",
            "meta_json": json.dumps(row.to_dict())
        }
        new_rows.append(entry)

# Create DataFrame
new_df = pd.DataFrame(new_rows)

# Remove duplicates
new_df.drop_duplicates(subset=["text"], inplace=True)

# Append to existing corpus
corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
corpus_df.reset_index(drop=True, inplace=True)
corpus_df["doc_id"] = corpus_df.index.astype(str)

# Save
corpus_df.to_pickle("final_corpus.pkl")

# Part 3
# Load existing corpus
corpus_df = pd.read_pickle("final_corpus.pkl")
start_id = corpus_df.shape[0]

# Load BioASQ dataset
bioasq_ds = load_from_disk("cp_datasets/bioasq_taskB_ALL")
bioasq_df = bioasq_ds.to_pandas()

def to_serializable(d):
    """Ensure all dict values are JSON serializable"""
    return {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in d.items()}

# Build formatted rows
new_rows = []
for row_idx, row in bioasq_df.iterrows():
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    snippets = row.get("snippets", [])

    if not question or not answer:
        continue

    # Extract actual text from snippets (list of dicts)
    snippet_block = "\n".join(s["text"].strip() for s in snippets if isinstance(s, dict) and "text" in s)

    full_text = f"Q: {question}\nA: {answer}"
    if snippet_block:
        full_text += f"\nSnippets:\n{snippet_block}"

    if not full_text.strip():
        continue

    new_rows.append({
        "doc_id": str(start_id + len(new_rows)),
        "text": full_text.strip(),
        "title": " ".join(question.split()[:10]),
        "source": "bioasq",
        "category": "summary",
        "meta_json": json.dumps(to_serializable(row.to_dict()))
    })

# Merge and deduplicate
new_df = pd.DataFrame(new_rows).drop_duplicates(subset=["text"])
corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
corpus_df.drop_duplicates(subset=["text"], inplace=True)

# Save
corpus_df.to_pickle("final_corpus.pkl")

# Part 4
# Load existing corpus
corpus_df = pd.read_pickle("final_corpus.pkl")
start_id = corpus_df.shape[0]

# Load Kaggle CSV
kaggle_df = pd.read_csv("cp_datasets/Kaggle_dataset_combined.csv")
# use: Dataset/symp_pre.csv

# Format new rows from Kaggle
new_rows = []
for idx, row in kaggle_df.iterrows():
    label = str(row.get("Label", "")).strip()
    definition = str(row.get("Def", "")).strip()
    title = str(row.get("Title", "")).strip()
    category = str(row.get("Category", "")).strip()

    if not label and not definition:
        continue

    text = f"{label}. {definition}".strip()

    new_rows.append({
        "doc_id": str(start_id + idx),
        "text": text,
        "title": title,
        "source": "kaggle",
        "category": category,
        "meta_json": json.dumps(row.to_dict(), default=str)
    })

# Deduplicate and merge
new_df = pd.DataFrame(new_rows).drop_duplicates(subset=["text"])
corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
corpus_df.drop_duplicates(subset=["text"], inplace=True)

# Save updated corpus
corpus_df.to_pickle("final_corpus.pkl")

# Check the medical corpus created

import pandas as pd
import matplotlib.pyplot as plt
# Load corpus
corpus_df = pd.read_pickle("final_corpus.pkl")
# Check for nulls
null_summary = corpus_df.isnull().sum()
print("🧪 Null Value Check:\n", null_summary)
# Column names
print("\n📋 Column Names:", corpus_df.columns.tolist())
# One sample
print("\n🔎 Sample Document:\n", corpus_df.sample(1).to_dict(orient="records")[0])
# Total samples
print(f"\n📦 Total documents in corpus: {len(corpus_df)}")

# 3. Create BM25_tokenized.pkl 
pip install rank-bm25 sentence-transformers torch numpy

# Torch Version: 2.0.1+cu118
# Transformers Version: 4.31.0

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import pickle
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the corpus we created above
corpus_path = "RAG_corpus/final_corpus.pkl"
corpus_df = pd.read_pickle(corpus_path)
texts = corpus_df["text"].astype(str).tolist()
print(f"📄 Loaded {len(texts)} texts")

# === BM25 Tokenized ===
print("🔍 Building BM25 tokenized index...")
tokenized = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized)
with open("RAG_corpus/bm25_tokenized.pkl", "wb") as f:
    pickle.dump(bm25, f) # Corrected: Save the bm25 object

# Create dense_embeddings.pt

tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
model.eval()

# encode embeddings 
batch_size = 32
embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="⚙️ Encoding"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded).last_hidden_state[:, 0, :]  # CLS
        embeddings.append(output.cpu())

all_embeds = torch.cat(embeddings, dim=0)
torch.save(all_embeds, "RAG_corpus/dense_embeddings.pt")
















