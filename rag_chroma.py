'''
Copyright (c) 2024 Ben Wang, Email: benwhz@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from typing import Any
from gradio_client import Client
import chromadb

# file reader
class SimpleFileReader():
    def __init__(self, filename) -> None:
        self.fn =filename
    
    def __call__(self) -> Any:
        with open(self.fn, 'r', encoding='utf-8') as file:
            return file.read()
    
# sentence splitter
class SentenceSplitter():
    def __init__(self, chunk_size = 512, chunk_overlap = 100) -> None:
        self.chunk = chunk_size
        self.overlap = chunk_overlap

    def split(self, docs):
        splits = []
        start = 0
        while True:
            end = start + self.chunk
            if end >= len(docs):
                end = len(docs)
            splits.append(docs[start:end])
            start = end - self.overlap
            if end == len(docs):
                break
        return splits

# chroma client
chroma_client = chromadb.Client()
collection_name = "rag_chroma_db"

try:
    chroma_client.delete_collection(name=collection_name)
except:
    pass
    
collection = chroma_client.create_collection(
      name=collection_name,
      metadata={"hnsw:space": "cosine"},
  )

# load text files
docs = SimpleFileReader('../dataset/news.txt')()
# normalize
docs = docs.replace("\r\n",'')
docs = docs.replace("\n",'')

splitter = SentenceSplitter()
chunks = splitter.split(docs)
#print(chunks[0])

try:
    collection.add(
        documents = chunks,
        ids=[f'id{n+1}' for n in range(len(chunks))]
    )
except:
    print('error')

# embedding model    
question = "Who will be the new Prime Minister of Japan?"
results = collection.query(
    query_texts=[question],
    n_results=1
)
print('results:', results['documents'][0][0])

# LLM from Huggingface
client = Client("Qwen/Qwen2.5")

# prompt with context.
query_text = "Consider the following background: " + results['documents'][0][0] + ", Please answer this question: " + question
print("Q: ", question)

result = client.predict(
		query=query_text,
		history=[],
		system="You are an AI robot. Please answer the questions according to the background information provided simply.",
		radio='72B',
		api_name="/model_chat"
)

answer_text = result[1][0][1]['text']
print("A: ", answer_text)

