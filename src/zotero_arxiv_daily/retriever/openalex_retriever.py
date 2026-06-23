import requests
from datetime import datetime
from omegaconf import ListConfig
from typing import Any

from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


OPENALEX_API_ROOT = "https://api.openalex.org"


def _normalize_strings(values: list[str] | ListConfig | None, config_key: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, ListConfig)):
        raise TypeError(f"config.source.openalex.{config_key} must be a list of strings or null.")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"config.source.openalex.{config_key} must contain only strings.")
        cleaned = value.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _normalize_keyword_groups(values: list[list[str]] | ListConfig | None, config_key: str) -> list[list[str]]:
    if values is None:
        return []
    if not isinstance(values, (list, ListConfig)):
        raise TypeError(f"config.source.openalex.{config_key} must be a list of string lists or null.")

    groups: list[list[str]] = []
    for group in values:
        if not isinstance(group, (list, ListConfig)):
            raise TypeError(f"config.source.openalex.{config_key} must contain only string lists.")
        normalized_group = _normalize_strings(group, config_key)
        if normalized_group:
            groups.append([item.lower() for item in normalized_group])
    return groups


def _parse_date(value: str | None, config_key: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError(f"config.source.openalex.{config_key} must be a string like '2024-01-01' or '2024/01'.")
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                parsed = parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"config.source.openalex.{config_key} must use YYYY-MM-DD, YYYY/MM/DD, YYYY-MM, or YYYY/MM format.")


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().split())


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _matches_groups(text: str, keyword_groups: list[list[str]]) -> bool:
    return all(any(keyword in text for keyword in group) for group in keyword_groups)


def _keyword_priority_score(text: str, priority_keywords: list[str], required_keyword_groups: list[list[str]]) -> int:
    score = sum(1 for keyword in priority_keywords if keyword in text)
    score += sum(sum(1 for keyword in group if keyword in text) for group in required_keyword_groups)
    return score


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        for offset in offsets:
            positions[offset] = word
    return " ".join(word for _, word in sorted(positions.items()))


@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.journals = _normalize_strings(self.retriever_config.get("journals"), "journals")
        self.conferences = _normalize_strings(self.retriever_config.get("conferences"), "conferences")
        self.source_ids = _normalize_strings(self.retriever_config.get("source_ids"), "source_ids")
        self.start_date = _parse_date(self.retriever_config.get("start_date"), "start_date")
        self.end_date = _parse_date(self.retriever_config.get("end_date"), "end_date")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("config.source.openalex.end_date must be later than or equal to start_date.")
        self.include_keywords = [item.lower() for item in _normalize_strings(self.retriever_config.get("include_keywords"), "include_keywords")]
        self.exclude_keywords = [item.lower() for item in _normalize_strings(self.retriever_config.get("exclude_keywords"), "exclude_keywords")]
        self.priority_keywords = [item.lower() for item in _normalize_strings(self.retriever_config.get("priority_keywords"), "priority_keywords")]
        self.required_keyword_groups = _normalize_keyword_groups(
            self.retriever_config.get("required_keyword_groups"),
            "required_keyword_groups",
        )
        self.candidate_pool_size = self.retriever_config.get("candidate_pool_size", 200)
        self.max_pages = self.retriever_config.get("max_pages", 3)
        self.per_page = self.retriever_config.get("per_page", 50)
        self.allowed_work_types = set(
            item.lower()
            for item in (_normalize_strings(self.retriever_config.get("work_types"), "work_types") or ["article", "proceedings-article"])
        )

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{OPENALEX_API_ROOT}{path}"
        response = requests.get(url, params=params, timeout=(10, 60))
        response.raise_for_status()
        return response.json()

    def _resolve_source_id(self, venue_name: str) -> tuple[str, str] | None:
        payload = self._request("/sources", {"search": venue_name, "per-page": 10})
        results = payload.get("results", [])
        normalized_target = _normalize_name(venue_name)

        exact_match = next(
            (
                source for source in results
                if _normalize_name(source.get("display_name", "")) == normalized_target
            ),
            None,
        )
        best_match = exact_match or (results[0] if results else None)
        if best_match is None:
            logger.warning(f"OpenAlex source not found for venue '{venue_name}'")
            return None
        return best_match["id"], best_match.get("display_name", venue_name)

    def _iter_venues(self) -> list[tuple[str, str]]:
        resolved: list[tuple[str, str]] = []
        seen_ids: set[str] = set()
        for source_id in self.source_ids:
            if source_id not in seen_ids:
                resolved.append((source_id, source_id))
                seen_ids.add(source_id)
        for venue_name in self.journals + self.conferences:
            match = self._resolve_source_id(venue_name)
            if match is None:
                continue
            source_id, display_name = match
            if source_id in seen_ids:
                continue
            resolved.append((source_id, display_name))
            seen_ids.add(source_id)
        if not resolved:
            raise ValueError("config.source.openalex requires at least one journal, conference, or source_id.")
        return resolved

    def _build_search_query(self) -> str | None:
        terms: list[str] = []
        for group in self.required_keyword_groups:
            if group:
                terms.append(group[0])
        terms.extend(self.include_keywords)
        terms.extend(self.priority_keywords)
        unique_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        if not unique_terms:
            return None
        return " ".join(unique_terms[:8])

    def _date_matches(self, publication_date: str | None) -> bool:
        if publication_date in (None, ""):
            return False
        try:
            paper_dt = datetime.strptime(publication_date, "%Y-%m-%d")
        except ValueError:
            return False
        if self.start_date and paper_dt < self.start_date:
            return False
        if self.end_date and paper_dt > self.end_date.replace(hour=23, minute=59, second=59):
            return False
        return True

    def _keyword_matches(self, text: str) -> bool:
        if self.required_keyword_groups and not _matches_groups(text, self.required_keyword_groups):
            return False
        if self.include_keywords and not _contains_any(text, self.include_keywords):
            return False
        if self.exclude_keywords and _contains_any(text, self.exclude_keywords):
            return False
        return True

    def _retrieve_for_venue(self, source_id: str, display_name: str) -> list[dict[str, Any]]:
        filters = [f"primary_location.source.id:{source_id}"]
        if self.start_date:
            filters.append(f"from_publication_date:{self.start_date.strftime('%Y-%m-%d')}")
        if self.end_date:
            filters.append(f"to_publication_date:{self.end_date.strftime('%Y-%m-%d')}")
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "sort": "publication_date:desc",
            "per-page": self.per_page,
            "cursor": "*",
        }
        search_query = self._build_search_query()
        if search_query:
            params["search"] = search_query

        results: list[dict[str, Any]] = []
        for page_index in range(self.max_pages):
            payload = self._request("/works", params)
            page_results = payload.get("results", [])
            logger.info(f"OpenAlex venue '{display_name}' page {page_index + 1}: {len(page_results)} raw works")
            if not page_results:
                break
            results.extend(page_results)
            next_cursor = payload.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
        return results

    def _prioritize(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def rank_key(paper: dict[str, Any]) -> tuple[int, str]:
            text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}".lower()
            priority_score = _keyword_priority_score(text, self.priority_keywords, self.required_keyword_groups)
            return priority_score, paper.get("publication_date") or ""

        papers = sorted(papers, key=rank_key, reverse=True)
        if self.candidate_pool_size:
            papers = papers[: self.candidate_pool_size]
        return papers

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        raw_papers: list[dict[str, Any]] = []
        for source_id, display_name in self._iter_venues():
            works = self._retrieve_for_venue(source_id, display_name)
            for work in works:
                publication_date = work.get("publication_date")
                work_type = (work.get("type") or "").lower()
                abstract = work.get("abstract") or _abstract_from_inverted_index(work.get("abstract_inverted_index"))
                text = f"{work.get('title', '')}\n{abstract}".lower()
                if work_type not in self.allowed_work_types:
                    continue
                if not self._date_matches(publication_date):
                    continue
                if not self._keyword_matches(text):
                    continue
                primary_location = work.get("primary_location") or {}
                source = primary_location.get("source") or {}
                work["venue_display_name"] = source.get("display_name") or display_name
                work["abstract"] = abstract
                raw_papers.append(work)

        logger.info(f"OpenAlex kept {len(raw_papers)} works after venue/date/keyword filtering")
        raw_papers = self._prioritize(raw_papers)
        logger.info(f"OpenAlex kept {len(raw_papers)} works after priority sorting and candidate pool limiting")
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        authorships = raw_paper.get("authorships") or []
        authors = [
            authorship.get("author", {}).get("display_name")
            for authorship in authorships
            if authorship.get("author", {}).get("display_name")
        ]
        doi = raw_paper.get("doi")
        primary_location = raw_paper.get("primary_location") or {}
        pdf_url = primary_location.get("pdf_url")
        landing_page_url = primary_location.get("landing_page_url") or raw_paper.get("id") or doi
        return Paper(
            source=self.name,
            title=raw_paper.get("title", "Untitled"),
            authors=authors,
            abstract=raw_paper.get("abstract", ""),
            url=landing_page_url,
            pdf_url=pdf_url or landing_page_url,
            full_text=None,
            venue=raw_paper.get("venue_display_name"),
            published_date=raw_paper.get("publication_date"),
            doi=doi,
        )
