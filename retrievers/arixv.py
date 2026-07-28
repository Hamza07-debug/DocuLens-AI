from __future__ import annotations

from typing import List

import arxiv
from langchain_core.documents import Document


def fetch_arxiv_documents(query: str, max_results: int = 3) -> List[Document]:
    """Fetch papers from arXiv and return them as LangChain Documents."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    docs: List[Document] = []
    for result in client.results(search):
        content_parts = [
            f"Title: {result.title}",
            f"Summary: {result.summary.strip()}",
        ]

        metadata = {
            "entry_id": result.entry_id,
            "published": result.published.isoformat() if result.published else None,
            "updated": result.updated.isoformat() if result.updated else None,
            "authors": [author.name for author in result.authors],
            "primary_category": result.primary_category,
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "doi": result.doi,
        }

        docs.append(
            Document(
                page_content="\n\n".join(content_parts),
                metadata=metadata,
            )
        )

    return docs


def main() -> None:
    query = "deep learning"
    try:
        docs = fetch_arxiv_documents(query=query, max_results=2)
    except Exception as exc:
        print(f"Failed to fetch arXiv data: {exc}")
        return

    if not docs:
        print("No documents found.")
        return

    for i, doc in enumerate(docs, start=1):
        print(f"Document {i}")
        print(doc.page_content)
        print(doc.metadata)
        print()


if __name__ == "__main__":
    main()

     

