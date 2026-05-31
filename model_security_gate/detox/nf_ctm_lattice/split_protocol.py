from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Any
import random


@dataclass(frozen=True)
class DisjointSplit:
    train: list[str]
    eval: list[str]
    unused: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_disjoint_split(items: Sequence[str], *, n_train: int, n_eval: int, seed: int = 20260526) -> DisjointSplit:
    """Create deterministic train/eval splits and raise on overlap risk."""
    uniq = list(dict.fromkeys(str(x) for x in items))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    if len(uniq) < int(n_train) + int(n_eval):
        raise ValueError(f"need at least {int(n_train)+int(n_eval)} unique items, got {len(uniq)}")
    train = sorted(uniq[: int(n_train)])
    eval_ = sorted(uniq[int(n_train): int(n_train) + int(n_eval)])
    unused = sorted(uniq[int(n_train) + int(n_eval):])
    if set(train) & set(eval_):
        raise RuntimeError("internal split error: train/eval overlap")
    return DisjointSplit(train=train, eval=eval_, unused=unused)
