from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        scores += self._keyword_bonus(candidates)
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

    def _keyword_bonus(self, candidates: list[Paper]) -> np.ndarray:
        if self.config is None:
            return np.zeros(len(candidates))
        arxiv_config = getattr(self.config.source, "arxiv", None)
        if arxiv_config is None:
            return np.zeros(len(candidates))

        priority_keywords = [kw.lower() for kw in (arxiv_config.get("priority_keywords") or []) if isinstance(kw, str)]
        if not priority_keywords:
            return np.zeros(len(candidates))

        bonuses = []
        for candidate in candidates:
            text = f"{candidate.title}\n{candidate.abstract}".lower()
            matches = sum(1 for keyword in priority_keywords if keyword in text)
            bonuses.append(matches * 0.35)
        return np.array(bonuses)
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
