import os
import bm25s
import json
import re
import time
from dotenv import load_dotenv


load_dotenv()

LAW_PATH = os.getenv("LAW_PATH")
K = int(os.getenv("K"))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def _pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

def _tokenize(text):
    return re.findall(r'\w+', text.lower())

def preprocess_articles(data):
    """Extracts and tokenizes content from legal articles."""
    articles = []
    article_info = []
    
    for doc in data:
        for chapter in doc["content"]:
            for section in chapter["content_Chapter"]:
                for article in section["content_Section"]:
                    article_title = _pre_processing(article["title_Article"])
                    content = _pre_processing(article["content_Article"])
                    tokenized_text = _tokenize(article_title + " " + content)
                    articles.append(tokenized_text)
                    article_info.append({
                        "doc_id": doc["id"],
                        "href": doc["href"],
                        "title": doc["title"],
                        "chapter_id": chapter["id_Chapter"],
                        "chapter_title": chapter["title_Chapter"],
                        "section_id": section["id_Section"],
                        "section_title": section["title_Section"],
                        "article_id": article["id_Article"],
                        "article_title": article["title_Article"],
                        "content": article["content_Article"]
                    })
    
    return articles, article_info


def _save_bm25s_retriever(retriever, filename='data/bm25s_retriever'):
    retriever.save(filename)

def _load_bm25s_retriever(filename = 'data/bm25s_retriever'):
    reloaded_retriever = bm25s.BM25.load(filename, load_corpus = False) # set load_corpus=False if you don't need the corpus
    return reloaded_retriever

def _save_article_info(article_info, filename="data/article_info.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(article_info, f, ensure_ascii=False, indent=4)

def _load_article_info(filename="data/article_info.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def bm25s_compute_docs(law_path):
    data = _load_json(law_path)
    articles, article_info = preprocess_articles(data)
    bm25s_retriever = bm25s.BM25() 
    bm25s_retriever.index(articles)
    _save_bm25s_retriever(bm25s_retriever, 'data/bm25s_retriever')
    _save_article_info(article_info)


def bm25s_retrieve(bm25s_retriever, article_info, query, k):
    """Returns full information of top-k relevant articles using BM25s scoring."""
    # Tokenize the query
    query_tokens = bm25s.tokenize(query)  

    # Retrieve top-k results
    results, scores = bm25s_retriever.retrieve(query_tokens, k=k)

    top_k_articles = []
    for i in range(results.shape[1]):
        doc_id = results[0, i]
        score = scores[0, i]
        article = article_info[doc_id].copy()
        article["bm25_score"] = score
        article["bm25_score_scaled"] = score / max(scores[0]) if max(scores[0]) > 0 else 0
        article["query"] = query
        top_k_articles.append(article)

    return top_k_articles


def BM25s(bm25s_retriever, article_info, query):
    start_time = time.time()  
    top_k_results = bm25s_retrieve(bm25s_retriever, article_info, query, K)
    end_time = time.time() 
    print(f"Done BM25s! Time taken: {end_time - start_time:.2f} seconds")
    return top_k_results


if __name__ == "__main__":
    bm25s_compute_docs(LAW_PATH)
