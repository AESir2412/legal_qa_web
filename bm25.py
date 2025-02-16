from constant import *

from rank_bm25 import BM25Okapi
import json
import re


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def preprocess_articles(data):
    """Extracts and tokenizes content from legal articles."""
    articles = []
    article_info = []
    
    for doc in data:
        for chapter in doc["content"]:
            for section in chapter["content_Chapter"]:
                for article in section["content_Section"]:
                    content = pre_processing(article["content_Article"])
                    tokenized_text = tokenize(content)
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
                        "content": article["content_Article"]
                    })
    
    return articles, article_info

def bm25_retrieve(query, articles, article_info, k=5):
    """Returns full information of top-k relevant articles using BM25 scoring."""
    tokenized_query = tokenize(query)
    bm25 = BM25Okapi(articles)
    scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_k_articles = []
    
    max_score = max(scores) if len(scores) > 0 else 1
    
    for i in ranked_indices:
        article = article_info[i].copy()
        article["bm25_score"] = scores[i]
        article["bm25_score_scaled"] = scores[i] / max_score if max_score > 0 else 0
        article["query"] = query
        top_k_articles.append(article)
    
    return top_k_articles


def bm25(query):
    data = load_json(LAW_PATH)
    articles, article_info = preprocess_articles(data)
    top_k_results = bm25_retrieve(query, articles, article_info, K)
    
    # with open("top_k_results.json", "w", encoding="utf-8") as f:
    #     json.dump(top_k_results, f, indent=2, ensure_ascii=False)

    print("Done BM25!")

    return top_k_results


if __name__ == "__main__":
    main()
    # bm25(DUMMY_QUERY)
