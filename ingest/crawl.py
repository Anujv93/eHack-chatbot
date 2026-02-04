import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

def get_site_urls(sitemap_url: str) -> list[str]:
    resp = requests.get(sitemap_url)
    soup = BeautifulSoup(resp.content, "xml")
    return [loc.text for loc in soup.find_all("loc")]

def load_pages(urls: list[str]):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs
