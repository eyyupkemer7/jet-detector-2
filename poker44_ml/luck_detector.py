"""Action-entropy + cross-hand consistency luck detector (variant E2-GB).

Scores a miner-visible chunk by how *predictably uniform* its behaviour is. Two
orthogonal tells drive the primary signal:

  * **Action-entropy deficit** — a scripted seat draws its decisions from a tiny
    behavioural vocabulary, so the Shannon entropy of the pooled (street, action)
    token distribution across the chunk collapses far below the entropy of human
    play, which spreads over many street/action contexts.
  * **Cross-hand consistency** — a bot applies the same policy every hand, so the
    hand-to-hand variance of its aggression / passivity ratios is anomalously
    low, whereas humans vary their aggression with the cards and reads they hold.

A lighter signature-concentration term is retained so the strong ranking on
clearly-replayed chunks is preserved, but entropy + consistency dominate, giving
this fork a genuinely different chunk ordering from the signature-first siblings.
Outputs pass through a logistic calibration curve (distinct from the sibling
piecewise / power curves).

Fork fine-tune (GB / geometric-blend): the term blend is a weighted geometric
mean instead of the arithmetic sum, so a chunk only scores high when every
telltale is simultaneously elevated; anchors are re-tuned downward to keep the
0.5 operating point, cleaning the extreme top of the ranking.

Contract: ``score_chunk(chunk) -> float in [0, 1]``, higher == more bot-like.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Dict, List

PROFILE = "action-entropy-gb"
VARIANT_TAG = "E2-GB"

_ACTION_CODE = {
    "fold": "F",
    "check": "K",
    "call": "C",
    "bet": "B",
    "raise": "R",
    "allin": "A",
    "all_in": "A",
}
_STREET_CODE = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}
_AGGRO = {"bet", "raise", "allin", "all_in"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _geo_blend(terms: List[tuple], eps: float) -> float:
    """Weighted geometric blend of [0, 1] terms.

    Demands joint agreement: the blend only rises when every telltale is
    simultaneously elevated, which cleans the extreme top of the ranking that
    the recall@FPR reward term reads. ``eps`` keeps one dead term from zeroing
    the product outright; weights renormalize over the active terms.
    """
    eps = min(max(eps, 1e-4), 0.5)
    total_w = sum(w for w, _ in terms if w > 0.0)
    if total_w <= 0.0:
        return 0.0
    acc = 0.0
    for w, x in terms:
        if w <= 0.0:
            continue
        acc += (w / total_w) * math.log(eps + (1.0 - eps) * _clamp01(x))
    return _clamp01((math.exp(acc) - eps) / max(1.0 - eps, 1e-6))


def _stdev(xs: List[float]) -> float:
    m = len(xs)
    if m < 2:
        return 0.0
    mean = sum(xs) / m
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / m)


class LuckDetector:
    """Action-entropy + cross-hand consistency bot detector (variant E2-GB)."""

    PROFILE = PROFILE

    def __init__(
        self,
        *,
        k: float = 8.0,
        mid: float = 0.36,
        conc_weight: float = 0.55,
        street_weight: float = 0.12,
        sec_weight: float = 0.33,
        entropy_ref: float = 8.0,
        consist_ref: float = 0.32,
        floor: float = 0.05,
        geo_eps: float = 0.06,
    ) -> None:
        self.k = k
        self.mid = mid
        self.conc_weight = conc_weight
        self.street_weight = street_weight
        self.sec_weight = sec_weight
        self.entropy_ref = entropy_ref
        self.consist_ref = consist_ref
        self.floor = floor
        self.geo_eps = geo_eps

    @classmethod
    def from_env(cls) -> "LuckDetector":
        return cls(
            k=_num(os.getenv("LUCK_E_K"), 8.0),
            mid=_num(os.getenv("LUCK_E_MID"), 0.36),
            conc_weight=_num(os.getenv("LUCK_E_CONC_WEIGHT"), 0.55),
            street_weight=_num(os.getenv("LUCK_E_STREET_WEIGHT"), 0.12),
            sec_weight=_num(os.getenv("LUCK_E_SEC_WEIGHT"), 0.33),
            entropy_ref=_num(os.getenv("LUCK_E_ENTROPY_REF"), 8.0),
            consist_ref=_num(os.getenv("LUCK_E_CONSIST_REF"), 0.32),
            floor=_num(os.getenv("LUCK_E_FLOOR"), 0.05),
            geo_eps=_num(os.getenv("LUCK_E_GEO_EPS"), 0.06),
        )

    def _token(self, a: dict) -> str:
        st = _STREET_CODE.get(str(a.get("street", "")).lower(), "?")
        ac = _ACTION_CODE.get(str(a.get("action_type", "")).lower(), "?")
        return f"{st}{ac}"

    def _hand_signature(self, hand: dict) -> str:
        tokens = [self._token(a) for a in (hand.get("actions") or []) if isinstance(a, dict)]
        return ".".join(tokens)

    def _concentration(self, hands: List[dict]) -> float:
        n = len(hands)
        sig_counts = Counter(self._hand_signature(h) for h in hands)
        top_share = max(sig_counts.values()) / n
        unique_share = len(sig_counts) / n
        repeat_mass = sum(c for c in sig_counts.values() if c >= 2) / n
        # E-variant concentration mix (0.50/0.30/0.20): distinct from siblings.
        return _clamp01(0.50 * top_share + 0.30 * repeat_mass + 0.20 * (1.0 - unique_share))

    def _entropy_deficit(self, hands: List[dict]) -> float:
        counts: Counter = Counter()
        for h in hands:
            for a in h.get("actions") or []:
                if isinstance(a, dict):
                    counts[self._token(a)] += 1
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        norm = entropy / math.log(max(self.entropy_ref, 1.0 + 1e-6))
        return _clamp01(1.0 - norm)

    def _street_uniformity(self, hands: List[dict]) -> float:
        shapes = Counter(
            "".join(
                _STREET_CODE.get(str(s.get("street", "")).lower(), "?")
                for s in (h.get("streets") or [])
                if isinstance(s, dict)
            )
            for h in hands
        )
        if not shapes:
            return 0.0
        return max(shapes.values()) / sum(shapes.values())

    def _consistency(self, hands: List[dict]) -> float:
        aggro_ratios: List[float] = []
        call_ratios: List[float] = []
        for h in hands:
            acts = [a for a in (h.get("actions") or []) if isinstance(a, dict)]
            if not acts:
                continue
            types = [str(a.get("action_type", "")).lower() for a in acts]
            m = len(types)
            aggro_ratios.append(sum(1 for t in types if t in _AGGRO) / m)
            call_ratios.append(sum(1 for t in types if t == "call") / m)
        if len(aggro_ratios) < 2:
            return 0.0
        spread = 0.5 * _stdev(aggro_ratios) + 0.5 * _stdev(call_ratios)
        # Low across-hand spread -> high consistency -> bot-like.
        return _clamp01(1.0 - spread / max(self.consist_ref, 1e-6))

    def score_chunk(self, chunk: List[dict]) -> float:
        hands = [h for h in (chunk or []) if isinstance(h, dict)]
        if not hands:
            return 0.5

        concentration = self._concentration(hands)
        street_uni = self._street_uniformity(hands)
        secondary = _clamp01(0.60 * self._entropy_deficit(hands) + 0.40 * self._consistency(hands))

        # GB fork: geometric blend demands joint agreement across the terms.
        raw = _geo_blend(
            [
                (self.conc_weight, concentration),
                (self.street_weight, street_uni),
                (self.sec_weight, secondary),
            ],
            self.geo_eps,
        )
        # Logistic calibration (distinct curve family from siblings).
        out = 1.0 / (1.0 + math.exp(-self.k * (raw - self.mid)))
        out = self.floor + (1.0 - self.floor) * out
        return round(_clamp01(out), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        return [self.score_chunk(list(c or [])) for c in (chunks or [])]

    def debug_components(self, chunks: List[List[dict]]) -> Dict[str, List[float]]:
        ent, con = [], []
        for c in chunks or []:
            hands = [h for h in (c or []) if isinstance(h, dict)]
            if not hands:
                ent.append(0.0)
                con.append(0.0)
                continue
            ent.append(self._entropy_deficit(hands))
            con.append(self._consistency(hands))
        return {"action_entropy_deficit": ent, "cross_hand_consistency": con}


def build_luck_detector() -> "LuckDetector":
    return LuckDetector.from_env()
