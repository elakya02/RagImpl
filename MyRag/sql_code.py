import numpy as np
import json
import sqlite3  
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


data_file = "Genie.json"
with open(data_file, 'r') as file:
    data = json.load(file) 
for value in data:
    embedding = model.encode(value['context'])
    value['encoded_matrix'] = embedding.tolist()

conn = sqlite3.connect('chunkdb.db')
curr = conn.cursor()

column_definitions = '"id" INTEGER, "topic" TEXT, "context" TEXT, "encoded_matrix" BLOB'
curr.execute('DROP TABLE IF EXISTS Chunking')
curr.execute(f'CREATE TABLE IF NOT EXISTS Chunking ({column_definitions})')

for item in data:
    embedding = item['encoded_matrix']
    byte_embed = np.array(embedding, dtype=np.float32).tobytes()
    values_to_insert = (
        item.get('id'),
        item.get('topic'),
        item.get('context'),
        byte_embed
    )
    curr.execute("INSERT INTO Chunking VALUES (?, ?, ?, ?)", values_to_insert)

conn.commit()
conn.close()