from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "vector_store/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    # Increased from k=3 to k=6 for broader context retrieval
    # This helps the bot give more comprehensive answers
    return vectorstore.as_retriever(search_kwargs={"k": 6})
