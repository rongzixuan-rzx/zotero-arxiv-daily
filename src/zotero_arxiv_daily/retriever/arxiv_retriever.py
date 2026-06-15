from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
from datetime import datetime
from omegaconf import ListConfig

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _normalize_keywords(values: list[str] | ListConfig | None, config_key: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, ListConfig)):
        raise TypeError(f"config.source.arxiv.{config_key} must be a list of strings or null.")
    keywords = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"config.source.arxiv.{config_key} must contain only strings.")
        keyword = value.strip().lower()
        if keyword:
            keywords.append(keyword)
    return keywords


def _normalize_keyword_groups(
    values: list[list[str]] | ListConfig | None,
    config_key: str,
) -> list[list[str]]:
    if values is None:
        return []
    if not isinstance(values, (list, ListConfig)):
        raise TypeError(f"config.source.arxiv.{config_key} must be a list of string lists or null.")

    groups: list[list[str]] = []
    for group in values:
        if not isinstance(group, (list, ListConfig)):
            raise TypeError(f"config.source.arxiv.{config_key} must contain only string lists.")
        normalized_group = _normalize_keywords(group, config_key)
        if normalized_group:
            groups.append(normalized_group)
    return groups


def _parse_date(value: str | None, config_key: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError(f"config.source.arxiv.{config_key} must be a string like '2024-01-01' or '2024/01'.")

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                parsed = parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"config.source.arxiv.{config_key} must use YYYY-MM-DD, YYYY/MM/DD, YYYY-MM, or YYYY/MM format.")


def _paper_datetime(paper: ArxivResult) -> datetime:
    paper_dt = getattr(paper, "published", None) or getattr(paper, "updated", None)
    if paper_dt is None:
        return datetime.min
    return paper_dt.replace(tzinfo=None)


def _paper_text(paper: ArxivResult) -> str:
    return f"{paper.title}\n{paper.summary}".lower()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _matches_groups(text: str, keyword_groups: list[list[str]]) -> bool:
    return all(any(keyword in text for keyword in group) for group in keyword_groups)


def _keyword_priority_score(text: str, priority_keywords: list[str], required_keyword_groups: list[list[str]]) -> int:
    score = sum(1 for keyword in priority_keywords if keyword in text)
    score += sum(sum(1 for keyword in group if keyword in text) for group in required_keyword_groups)
    return score


def _query_term(term: str) -> str:
    escaped = term.replace('"', "")
    return f'all:"{escaped}"'


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        self.categories = set(self.config.source.arxiv.category)
        self.include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        self.start_date = _parse_date(self.retriever_config.get("start_date"), "start_date")
        self.end_date = _parse_date(self.retriever_config.get("end_date"), "end_date")
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("config.source.arxiv.end_date must be later than or equal to start_date.")
        self.include_keywords = _normalize_keywords(self.retriever_config.get("include_keywords"), "include_keywords")
        self.exclude_keywords = _normalize_keywords(self.retriever_config.get("exclude_keywords"), "exclude_keywords")
        self.priority_keywords = _normalize_keywords(self.retriever_config.get("priority_keywords"), "priority_keywords")
        self.required_keyword_groups = _normalize_keyword_groups(
            self.retriever_config.get("required_keyword_groups"),
            "required_keyword_groups",
        )
        self.candidate_pool_size = self.retriever_config.get("candidate_pool_size")
        self.max_results = self.retriever_config.get("max_results")

    def _build_query(self) -> str:
        clauses = [f"({' OR '.join(f'cat:{category}' for category in sorted(self.categories))})"]
        if self.start_date or self.end_date:
            start = (self.start_date or datetime(1991, 1, 1)).strftime("%Y%m%d0000")
            end = (self.end_date or datetime.utcnow()).strftime("%Y%m%d2359")
            clauses.append(f"submittedDate:[{start} TO {end}]")
        for group in self.required_keyword_groups:
            clauses.append(f"({' OR '.join(_query_term(keyword) for keyword in group)})")
        if self.include_keywords:
            clauses.append(f"({' OR '.join(_query_term(keyword) for keyword in self.include_keywords)})")
        return " AND ".join(clauses)

    def _category_matches(self, paper: ArxivResult) -> bool:
        paper_categories = set(getattr(paper, "categories", []) or [])
        if self.include_cross_list:
            return bool(self.categories & paper_categories)
        return getattr(paper, "primary_category", None) in self.categories

    def _date_matches(self, paper: ArxivResult) -> bool:
        paper_dt = _paper_datetime(paper)
        if self.start_date and paper_dt < self.start_date:
            return False
        if self.end_date and paper_dt > self.end_date.replace(hour=23, minute=59, second=59):
            return False
        return True

    def _keyword_matches(self, paper: ArxivResult) -> bool:
        text = _paper_text(paper)
        if self.required_keyword_groups and not _matches_groups(text, self.required_keyword_groups):
            return False
        if self.include_keywords and not _contains_any(text, self.include_keywords):
            return False
        if self.exclude_keywords and _contains_any(text, self.exclude_keywords):
            return False
        return True

    def _prioritize(self, papers: list[ArxivResult]) -> list[ArxivResult]:
        def rank_key(paper: ArxivResult) -> tuple[int, datetime]:
            text = _paper_text(paper)
            priority_score = _keyword_priority_score(text, self.priority_keywords, self.required_keyword_groups)
            return priority_score, _paper_datetime(paper)

        papers = sorted(papers, key=rank_key, reverse=True)
        if self.candidate_pool_size:
            papers = papers[: self.candidate_pool_size]
        return papers

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = self._build_query()
        logger.info(f"Using arXiv query: {query}")
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        raw_papers = list(client.results(search))
        logger.info(f"Retrieved {len(raw_papers)} raw arXiv hits before filtering")
        raw_papers = [paper for paper in raw_papers if self._category_matches(paper)]
        logger.info(f"{len(raw_papers)} papers remain after category filtering")
        raw_papers = [paper for paper in raw_papers if self._date_matches(paper)]
        logger.info(f"{len(raw_papers)} papers remain after date filtering")
        raw_papers = [paper for paper in raw_papers if self._keyword_matches(paper)]
        logger.info(f"{len(raw_papers)} papers remain after keyword filtering")
        raw_papers = self._prioritize(raw_papers)
        logger.info(f"{len(raw_papers)} papers remain after priority sorting and candidate pool limiting")
        if self.config.executor.debug:
            raw_papers = raw_papers[:10]
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
