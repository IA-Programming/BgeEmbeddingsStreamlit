import os
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st

st.set_page_config(
    page_title="Trying to see embeddings💬", page_icon="🤗", layout="wide", initial_sidebar_state="expanded"
)

st.title('🤗💬 Embeddings BGE')

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

## Here is the nmew embeddings being used
embedding = model_norm

upload_pdf = st.file_uploader("Subir tu DOCUMENTO", type=['txt', 'pdf'], accept_multiple_files=True)
if upload_pdf is not None and st.button('📝✅ Cargar Documentos'):
    documents = []
    with st.spinner('🔨 Leyendo documentos...'):
        for upload_pdf in upload_pdf:
            print(upload_pdf.type)
            if upload_pdf.type == 'text/plain':
                documents += [upload_pdf.read().decode()]
            elif upload_pdf.type == 'application/pdf':
                with pdfplumber.open(upload_pdf) as pdf:
                    documents += [page.extract_text() for page in pdf.pages]

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)

        db = FAISS.from_documents(docs, embedding)

    st.write(docs)

    # if prompt:=st.text_input("Insert your query here"):
    st.write(db.as_retriever("interests"))
