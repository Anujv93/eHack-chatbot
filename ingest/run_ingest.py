import os
from crawl import get_site_urls, load_pages
from chunk import chunk_documents
from build_index import build_faiss
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# ── 1. Crawl all website pages from sitemap ────────────────────────
print("📡 Crawling website pages...")
urls = get_site_urls("https://www.ehackacademy.com/sitemap.xml")
docs = load_pages(urls)
print(f"   Loaded {len(docs)} pages from sitemap")

# ── 2. Load local knowledge base document ─────────────────────────
KB_PATH = os.path.join(
    os.path.dirname(__file__),  # ingest/
    "..", "rag_service", "storage", "ehack_context.md"
)
KB_PATH = os.path.abspath(KB_PATH)

if os.path.exists(KB_PATH):
    print(f"📄 Loading knowledge base: {KB_PATH}")
    kb_loader = UnstructuredMarkdownLoader(KB_PATH)
    kb_docs = kb_loader.load()
    # Tag the source so it can be identified in retrieval
    for doc in kb_docs:
        doc.metadata["source"] = "ehack_knowledge_base"
    docs.extend(kb_docs)
    print(f"   Loaded {len(kb_docs)} knowledge base chunks")
else:
    print(f"⚠️  Knowledge base not found at {KB_PATH} — skipping")

# ── 3. Chunk all documents ─────────────────────────────────────────
print("✂️  Chunking documents...")
chunks = chunk_documents(docs)
print(f"   Created {len(chunks)} chunks total")

# ── 4. Build FAISS index ───────────────────────────────────────────
print("🔨 Building FAISS index...")
build_faiss(chunks)

print("✅ Ingestion complete — vector store updated")
