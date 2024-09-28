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
from sentence_transformers import SentenceTransformer
from gradio_client import Client

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

# vector store
class VectorStoreIndex():
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.vectors = {}
    
    def __call__(self, chunks) -> Any:
        for i, chunk in enumerate(chunks):
            embeddings = self.embedding.encode(chunk)
            self.vectors[i] = (chunk, embeddings)
        print('vector length = ', len(self.vectors))
        return self
    
    def query(self, query, top_k = 1):
        q_emb = self.embedding.encode(query)
        max_i = 0
        max_value = 0
        for i, value in self.vectors.items():
            similarity = q_emb @ value[1].T
            if similarity > max_value:
                max_value = similarity
                max_i = i
        # print(max_i, max_value)
        return (max_i, self.vectors[max_i][0])
    
    def clear(self):
        self.vectors.clear()
    
docs = SimpleFileReader('../dataset/news.txt')()
# normalize
docs = docs.replace("\r\n",'')
docs = docs.replace("\n",'')

splitter = SentenceSplitter()
chunks = splitter.split(docs)

# embedding model    
embedding = SentenceTransformer('BAAI/bge-small-zh-v1.5')
index = VectorStoreIndex(embedding)(chunks)

question = "Who will be the new Prime Minister of Japan?"
_, context = index.query(question)

# LLM from Huggingface
client = Client("Qwen/Qwen2.5")

# prompt with context.
query_text = "Consider the following background: " + context + ", Please answer this question: " + question
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

