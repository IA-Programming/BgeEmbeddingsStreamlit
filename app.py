from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st

st.set_page_config(
    page_title="Trying to see embeddings💬", page_icon="🤗", layout="wide", initial_sidebar_state="expanded"
)

st.title('🤗💬 Embeddings BGE')

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

loader = TextLoader("/content/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

## Here is the nmew embeddings being used
embedding = model_norm

db = FAISS.from_documents(docs, embedding)

query = "What did the president say about Ketanji Brown Jackson"
Texts = db.similarity_search(query)

st.write(Texts[0].page_content)