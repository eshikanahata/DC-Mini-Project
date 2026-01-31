import pandas as pd
from sentence_transformers import SentenceTransformer
import csv

words = [
    # 5 fruits
    "apple", "banana", "orange", "grape", "mango",
    # tech companies
    "google", "microsoft", "apple", "tesla", "nvidia",
    # 5 emotions
    "joy", "sadness", "anger", "fear", "surprise",
    # 5 random objects
    "chair", "table", "laptop", "desk", "coffee mug"
]

# 2 : load the model
print("loading model : ")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3 : generate embeddings
print("generating embeddings : ")
embeddings = model.encode(words)

# 4 : save to tsv files
df_vectors = pd.DataFrame(embeddings)
df_vectors.to_csv('vectors.tsv', sep='\t', index=False, header=False)

# save labels
df_metadata = pd.DataFrame(words, columns=['Word'])
df_metadata.to_csv('metadata.tsv', sep='\t', index=False, header=False)

print("Generated and saved vectors.tsv and metadata.tsv")
