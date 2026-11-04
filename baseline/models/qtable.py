"""Bidirectional dynamic Q-tables for a naming-game population."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from baseline.data import Alphabet


class BidirectionalQTable:
    def __init__(self, alpha: float = 0.5, init_score: float = 0.5):
        self.alpha = alpha
        self.init_score = init_score
        self.q: Dict[Tuple[str, str], float] = {}
        self.words_for: Dict[str, Set[str]] = {}
        self.objects_for: Dict[str, Set[str]] = {}

    def _ensure(self, obj: str, word: str) -> None:
        key = (obj, word)
        if key not in self.q:
            self.q[key] = self.init_score
            self.words_for.setdefault(obj, set()).add(word)
            self.objects_for.setdefault(word, set()).add(obj)

    def get(self, obj: str, word: str) -> float:
        return self.q.get((obj, word), self.init_score)

    def update(self, obj: str, word: str, reward: float) -> float:
        self._ensure(obj, word)
        old = self.q[(obj, word)]
        new = old + self.alpha * (reward - old)
        self.q[(obj, word)] = new
        return abs(new - old)

    def best_word(self, obj: str) -> Optional[str]:
        words = self.words_for.get(obj)
        if not words:
            return None
        return max(words, key=lambda w: self.get(obj, w))

    def objects_for_word(self, word: str) -> Set[str]:
        return self.objects_for.get(word, set())

    def competing_words(self, obj: str, used: str) -> List[str]:
        return [w for w in self.words_for.get(obj, set()) if w != used]

    @property
    def lexicon_size(self) -> int:
        return sum(1 for v in self.q.values() if v >= 0.01)


class QPopulation:
    def __init__(
        self,
        n_agents: int,
        concept_names: List[str],
        alphabet: Alphabet,
        max_length: int,
        alpha: float,
        init_score: float,
        seed: int,
    ):
        self.agents = [BidirectionalQTable(alpha=alpha, init_score=init_score) for _ in range(n_agents)]
        self.names = concept_names
        self.alphabet = alphabet
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.used_forms: Set[str] = set()

    def invent(self) -> str:
        for _ in range(1000):
            word = self.alphabet.random_word(self.max_length, self.rng)
            if word not in self.used_forms:
                self.used_forms.add(word)
                return word
        word = self.alphabet.random_word(self.max_length, self.rng) + f"-{len(self.used_forms)}"
        self.used_forms.add(word)
        return word

    def speak(self, agent: BidirectionalQTable, obj: str) -> str:
        word = agent.best_word(obj)
        if word is None:
            word = self.invent()
            agent._ensure(obj, word)
        return word

    def listen(
        self,
        agent: BidirectionalQTable,
        word: str,
        candidate_names: List[str],
    ) -> int:
        mapped = agent.objects_for_word(word)
        in_context = [i for i, n in enumerate(candidate_names) if n in mapped]
        if not in_context:
            return self.rng.randrange(len(candidate_names))
        return max(in_context, key=lambda i: agent.get(candidate_names[i], word))

    def align(
        self,
        speaker: BidirectionalQTable,
        listener: BidirectionalQTable,
        obj: str,
        word: str,
        success: bool,
        listener_obj: Optional[str],
    ) -> float:
        delta = 0.0
        if success:
            delta += speaker.update(obj, word, 1.0)
            delta += listener.update(obj, word, 1.0)
            for other in speaker.competing_words(obj, word):
                delta += speaker.update(obj, other, 0.0)
            for other in listener.competing_words(obj, word):
                delta += listener.update(obj, other, 0.0)
        else:
            delta += speaker.update(obj, word, 0.0)
            if listener_obj is not None and (listener_obj, word) in listener.q:
                delta += listener.update(listener_obj, word, 0.0)
        return delta

    def mean_lexicon_size(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.lexicon_size for a in self.agents) / len(self.agents)

    def state_dict(self) -> dict:
        return {
            "q": [dict(a.q) for a in self.agents],
            "alpha": self.agents[0].alpha if self.agents else 0.5,
            "init_score": self.agents[0].init_score if self.agents else 0.5,
            "used_forms": list(self.used_forms),
        }

    def load_state_dict(self, state: dict) -> None:
        alpha = state.get("alpha", 0.5)
        init_score = state.get("init_score", 0.5)
        self.used_forms = set(state.get("used_forms", []))
        tables = state["q"]
        self.agents = []
        for qdict in tables:
            agent = BidirectionalQTable(alpha=alpha, init_score=init_score)
            for (obj, word), value in qdict.items():
                agent.q[(obj, word)] = value
                agent.words_for.setdefault(obj, set()).add(word)
                agent.objects_for.setdefault(word, set()).add(obj)
            self.agents.append(agent)
