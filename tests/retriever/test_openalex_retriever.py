from types import SimpleNamespace

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.openalex_retriever import OpenAlexRetriever


def test_openalex_retriever_filters_and_converts(config, monkeypatch):
    with open_dict(config.source):
        config.source.openalex = {
            "journals": ["IEEE Transactions on Software Engineering"],
            "conferences": [],
            "source_ids": None,
            "start_date": "2024-01-01",
            "end_date": None,
            "include_keywords": None,
            "required_keyword_groups": [
                ["smart contract", "solidity", "evm"],
                ["vulnerability", "security", "audit", "detection"],
            ],
            "exclude_keywords": ["malware"],
            "priority_keywords": ["smart contract vulnerability", "solidity"],
            "candidate_pool_size": 10,
            "max_pages": 1,
            "per_page": 10,
            "work_types": ["article", "proceedings-article"],
        }

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/sources"):
            payload = {
                "results": [
                    {
                        "id": "https://openalex.org/S123",
                        "display_name": "IEEE Transactions on Software Engineering",
                    }
                ]
            }
        elif url.endswith("/works"):
            payload = {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "Smart Contract Vulnerability Detection via Symbolic Execution",
                        "type": "article",
                        "publication_date": "2025-05-01",
                        "doi": "https://doi.org/10.1234/example",
                        "abstract_inverted_index": {
                            "smart": [0],
                            "contract": [1],
                            "vulnerability": [2],
                            "detection": [3],
                        },
                        "primary_location": {
                            "landing_page_url": "https://example.com/paper",
                            "pdf_url": "https://example.com/paper.pdf",
                            "source": {"display_name": "IEEE Transactions on Software Engineering"},
                        },
                        "authorships": [{"author": {"display_name": "Alice"}}],
                    },
                    {
                        "id": "https://openalex.org/W2",
                        "title": "Malware Detection with Deep Learning",
                        "type": "article",
                        "publication_date": "2025-05-01",
                        "doi": None,
                        "abstract": "malware detection",
                        "primary_location": {
                            "landing_page_url": "https://example.com/malware",
                            "pdf_url": None,
                            "source": {"display_name": "IEEE Transactions on Software Engineering"},
                        },
                        "authorships": [{"author": {"display_name": "Bob"}}],
                    },
                ],
                "meta": {"next_cursor": None},
            }
        else:
            raise AssertionError(f"Unexpected URL {url}")

        response = SimpleNamespace()
        response.raise_for_status = lambda: None
        response.json = lambda: payload
        return response

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    retriever = OpenAlexRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 1
    assert papers[0].title.startswith("Smart Contract Vulnerability Detection")
    assert papers[0].venue == "IEEE Transactions on Software Engineering"
    assert papers[0].published_date == "2025-05-01"
