from typing import Any
from sentence_transformers import SentenceTransformer
from gradio_client import Client, handle_file

# embedding model    
embedding = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# file reader
class SimpleFileReader():
    def __init__(self, filename) -> None:
        self.fn =filename
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
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
        exit = False
        while True:
            end = start + self.chunk
            if end >= len(docs):
                end = len(docs)
                exit = True
            splits.append(docs[start:end])
            start = end - self.overlap
            if exit:
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
splitter = SentenceSplitter()
chunks = splitter.split(docs)
index = VectorStoreIndex(embedding)(chunks)

question = "Who will be the new Prime Minister of Japan?"
_, context = index.query(question)
# print(context)

# LLM from Huggingface
client = Client("Qwen/Qwen2.5")

# prompt with context.
query_text = "Consider the following background: " + context + ", Please answer this question: " + question

result = client.predict(
		query=query_text,
		history=[],
		system="You are an AI robot. Please answer the questions according to the background information provided.",
		radio='72B',
		api_name="/model_chat"
)

print(result)

exit()

#print(len(splits), splits[-2], splits[-1])
embeddings_2 = embedding.encode(chunks[0])
embeddings_1 = embedding.encode(chunks[-1])
print(type(embeddings_1), embeddings_1.shape, embeddings_2.shape)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)