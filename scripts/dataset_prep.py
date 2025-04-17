# Check paths carefully

### 1. MedQuAD dataset 
# clone repo: !git clone https://github.com/abachaa/MedQuAD.git

# View the repo
import os

base_dir = "MedQuAD"

print("📁 Top-level contents of MedQuAD folder:")
for item in os.listdir(base_dir):
    print(" -", item)

# extract data and save in a single JSON file

import os
import json
import xml.etree.ElementTree as ET

# Folder containing all 12 MedQuAD subfolders
medquad_root = "cp_datasets/MedQuAD"
output_path = "cp_datasets/medquad_corpus/medquad_corpus.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

qa_pairs = []
skipped = 0

def extract_qa_from_xml(file_path):
    global skipped
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for qa in root.findall(".//QAPair"):
            q_node = qa.find("Question")
            a_node = qa.find("Answer")
            question = q_node.text.strip() if q_node is not None and q_node.text else None
            answer = a_node.text.strip() if a_node is not None and a_node.text else None
            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
            else:
                skipped += 1
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        skipped += 1

# Walk through all subfolders and XML files
for root, dirs, files in os.walk(medquad_root):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root, file)
            extract_qa_from_xml(file_path)

# Save extracted corpus to disk
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

# Summary
print(f"\n✅ Extracted {len(qa_pairs)} Q/A pairs.")
print(f"❌ Skipped {skipped} malformed or incomplete entries.")
print(f"💾 Saved cleaned corpus to: {output_path}")


# MedQuAD columns: ['question','answer']

### 2. BioASQ_taskB dataset
# download dataset from here: https://participants-area.bioasq.org/datasets/
# file will be downloaded in JSON format where, BioASQ Columns: ['body', 'documents', 'ideal_answer', 'concepts', 'type', 'id', 'snippets']

#process and edit columns of BioASQ:

import os
import json
import pandas as pd
from datasets import Dataset

# Define dataset paths
data_dir = "cp_datasets"
bioasq_path = os.path.join(data_dir, "BioASQ_taskB.json")

# Load BioASQ dataset
with open(bioasq_path, "r") as f:
    bioasq_data = json.load(f)

# Preprocess BioASQ dataset
processed_bioasq = []
for qa in bioasq_data["questions"]:
    question = qa["body"]  # Rename body → question
    answer = qa.get("ideal_answer", ["N/A"])  # Use ideal_answer, default to "N/A"
    if isinstance(answer, list):
        answer = "; ".join(answer)  # Convert list to string
    processed_bioasq.append({"question": question, "answer": answer, "type": qa["type"]})

# Original BioASQ Columns: ['body', 'documents', 'ideal_answer', 'concepts', 'type', 'id', 'snippets']
# 'body' -> 'question', 'ideal_answer'-> 'answer', remove 'id', 'concepts'
# processed BioASQ_taskB columns: ['question', 'answer', 'type', 'documents', 'snippets']

### 3. Symptoms & precautions created dataset from Kaggle
# file is uploaded here in CSV format - path: Dataset/Kaggle_dataset_combined.csv
# Column names: ['Label', 'Def', 'Title', 'Source', 'Category']

### 4. MedMCQA dataset
# load dataset from Hugging Face

import datasets

def load_medmcqa_dataset():
 
  try:
    dataset = datasets.load_dataset("medmcqa")
    return dataset
  except Exception as e:
    print(f"Error loading MedMCQA dataset: {e}")
    return None

if __name__ == "__main__":
  medmcqa = load_medmcqa_dataset()
  if medmcqa:
    print("MedMCQA dataset loaded successfully!")
    print(medmcqa)  # Print dataset information
    # You can now access the data like this:
    # print(medmcqa['train'][0])
  else:
    print("Failed to load MedMCQA dataset.")


#  Column: ['id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name']

#process columns

from datasets import load_dataset, Dataset
import pandas as pd
import torch
import os
from datetime import datetime


# Create output directory
output_dir = "cp_datasets/medmcqa_processed"
os.makedirs(output_dir, exist_ok=True)

# Load MedMCQA dataset
print("Loading MedMCQA dataset...")
dataset = load_dataset('medmcqa')

# Function to process data into the desired format
def process_data(data_items):
    processed_data = {
        'context': [],
        'question': [],
        'answer': []
    }
    
    for item in data_items:
        # Extract context (handle None values)
        context = item['exp'] if item['exp'] is not None else ""
        
        # Extract question
        question = item['question']
        
        # Extract answer - get the correct option based on 'cop'
        options = [item['opa'], item['opb'], item['opc'], item['opd']]
        cop_index = int(item['cop']) - 1  # Convert 1-based index to 0-based
        answer = options[cop_index]
        
        # Add to processed data
        processed_data['context'].append(context)
        processed_data['question'].append(question)
        processed_data['answer'].append(answer)
    
    return processed_data

# Combine all splits into a single dataset
print("Combining all splits into a single dataset...")
all_data_items = []
for split_name in dataset.keys():
    all_data_items.extend(dataset[split_name])
    print(f"Added {len(dataset[split_name])} examples from {split_name} split")

# Process all data
print("Processing combined dataset...")
processed_data = process_data(all_data_items)
print(f"Total processed examples: {len(processed_data['question'])}")

# Convert to a Hugging Face dataset
print("Converting to HuggingFace dataset...")
hf_dataset = Dataset.from_dict(processed_data)

# Save the processed dataset
print(f"Saving processed dataset to {output_dir}...")
hf_dataset.save_to_disk(output_dir)

#  Column: ['id', 'question', 'opa', 'opb', 'opc', 'opd', 'cop', 'choice_type', 'exp', 'subject_name', 'topic_name']
# After processing final columns, we get are: ['context', 'question', 'answer']
# From the original column, we took 'question' as 'question and 'exp' as 'context'
# How 'answer' was created: from 'cop' the correct option is triggered and correct word is extracted and saved in 'answer'. 






















































