from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    # Increased chunk_size from 500 to 1200 and overlap from 20 to 200
    # This keeps related information together (e.g., program name + fees + EMI + details)
    # and prevents critical data from being split across chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n---\n", "\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(docs)
