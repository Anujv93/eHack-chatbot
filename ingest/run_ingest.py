from crawl import get_site_urls, load_pages
from chunk import chunk_documents
from build_index import build_faiss

urls = get_site_urls("https://www.ehackacademy.com/sitemap.xml")
docs = load_pages(urls)
chunks = chunk_documents(docs)

build_faiss(chunks)
print("✅ Ingestion complete")
