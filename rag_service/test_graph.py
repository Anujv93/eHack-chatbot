from rag_service.graph import app

def run():
    print("🧪 Testing RAG Graph locally (no API)...")
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break

        result = app.invoke({"query": q})
        print("\n🤖 Answer:\n", result["answer"])

if __name__ == "__main__":
    run()
