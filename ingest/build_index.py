from langchain_community.vectorstores import FAISS
from .embed import get_embeddings

def build_faiss(chunks, path="vector_store/faiss_index"):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(path)
