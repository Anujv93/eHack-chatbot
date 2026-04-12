import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from crawl import get_site_urls, load_pages
from chunk import chunk_documents
from build_index import build_faiss
from langchain_core.documents import Document

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
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_text = f.read()

    # Split the markdown by section headers (##) to create logical documents
    sections = kb_text.split("\n## ")
    kb_docs = []
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        # Re-add the ## prefix (except for the first section which has the title)
        content = section if i == 0 else f"## {section}"
        kb_docs.append(Document(
            page_content=content,
            metadata={"source": "ehack_knowledge_base", "section": i}
        ))

    docs.extend(kb_docs)
    print(f"   Loaded {len(kb_docs)} knowledge base sections")
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
