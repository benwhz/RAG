# RAG

We built a simple RAG (Retrieval-augmented generation) pipeline from scratch, using the following open sources models on [huggingface](https://huggingface.co/):
- Embedding: [FlagEmbedding](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- LLM: [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)

**RAG modules**:
1. SimpeFileReader: just read a text file
1. SentenceSplitter: split the docs with specified chunk size and overlap.
1. VectorStoreIndex: build a vector store with the chunked nodes and create the index.

**Work steps**:
1. read the text file.
1. normalize the file content.
1. split the content into chunks.
1. build the vector store with the chunks.
1. define the question.
1. query the context of this quesion from vector store.
1. format the prompt with the question and it's context.
1. predict the prompt with the LLM.
1. get the answer.

Tested and run under Python 3.12.6, the following packages are also required
- [sentence-transformers 3.1.1](https://www.sbert.net/)
- [gradio-client 1.3.0](https://www.gradio.app/guides/getting-started-with-the-python-client)

> The test text documents were obtained from Google search on the Internet.