from sentence_transformers import SentenceTransformer
import csv

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 20 words across 4 categories
words = [
    # Fruits
    "apple", "banana", "mango", "orange", "grape",

    # Tech companies
    "google", "microsoft", "amazon", "meta", "apple_inc",

    # Emotions
    "happy", "sad", "angry", "excited", "calm",

    # Objects
    "chair", "car", "phone", "bottle", "backpack"
]

# Generate embeddings
embeddings = model.encode(words)

# Save vectors
with open("vectors.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for vector in embeddings:
        writer.writerow(vector)

# Save labels
with open("metadata.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for word in words:
        writer.writerow([word])

print("vectors.tsv and metadata.tsv created")
