import logging
import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]


class Pretokenizer(ABC):
    """
    Performs pre-tokenization on the text corpus in the given file.
    """

    @abstractmethod
    def run(self, input_path: str | os.PathLike, special_tokens: Sequence[str]) -> Counter[Tokens]:
        pass


class NaivePretokenizer(Pretokenizer):
    """
    Performs the simple example pre-tokenization as described in the BPE training example,
    with no buffering, parallelization, and only splitting on whitespace.
    """

    def run(self, input_path: str | os.PathLike, special_tokens: Sequence[str]) -> Counter[Tokens]:
        raw = Path(input_path).read_bytes()
        pre_tokenized = raw.split()
        logger.info("Read %d bytes, split into %d whitespace tokens", len(raw), len(pre_tokenized))
        tokenized = (tuple(word[i : i + 1] for i in range(len(word))) for word in pre_tokenized)
        tokenized_count: Counter[tuple[bytes, ...]] = Counter(tokenized)
        logger.info("Character-tokenized: %d unique token sequences", len(tokenized_count))

        return tokenized_count
