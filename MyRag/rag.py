import urllib3
import ssl
import certifi
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import json

converter = DocumentConverter()
result = converter.convert("protocol.pdf")

doc = result.document

#Converting document into chunks
import re
import json
chunker = HybridChunker()
chunk_iter = chunker.chunk(dl_doc=doc)

result_list = []
my_json = {}
j = 1
for i, chunk in enumerate(chunk_iter, 1):
    enriched_text = chunker.contextualize(chunk=chunk)
    my_json = {}
    # with open("Chunky.txt" , 'a') as file:
    #   file.write(f"{enriched_text}\n")
    # print(enriched_text)
    key = enriched_text.split('\n', 1)[0]
    value = enriched_text.replace(key , "")
    # print(key)
    # print(value)
    my_json["id"] = j
    my_json["topic"] = key
    my_json["context"] = value
    result_list.append(my_json)
    # print(my_json)
    # print(result_list)
    j = j+1


output_filename = "Genie.json"

with open(output_filename, 'w') as file:
    json.dump(result_list, file, indent=4)
