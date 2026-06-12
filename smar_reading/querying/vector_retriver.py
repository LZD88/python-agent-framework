from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def recall_with_scores(
        query: str,
        vectorstore: Chroma,
        k: int = 30
) -> Tuple[List[Document], List[float]]:
    # similarity_search_with_relevance_scores 返回 List[Tuple[Document, float]]
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    docs = []
    scores = []
    for doc, score, in results:
        doc.page_content = doc.page_content.strip()
        docs.append(doc)
        scores.append(score)

    return docs, scores
