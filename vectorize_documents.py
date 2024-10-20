import os
import langchain_community
import langchain_text_splitters
import langchain_huggingface
import langchain_chroma

from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# loading the embedding model
embeddings = HuggingFaceEmbeddings()

loader = JSONLoader(file_path='./train-v2.0.json', text_content=False, jq_schema='.data[].paragraphs[].context')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000,
                                      chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized")
