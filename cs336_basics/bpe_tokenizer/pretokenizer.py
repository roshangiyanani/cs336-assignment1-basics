import logging
import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]


class Pretokenizer(ABC):
    """
    Performs pre-tokenization on the text corpus in the given file.
    """

    @abstractmethod
    def process(self, segment: bytes) -> None:
        pass

    @abstractmethod
    def finalize(self) -> Counter[Tokens]:
        pass


class NaivePretokenizer(Pretokenizer):
    """
    Performs the simple example pre-tokenization as described in the BPE training example,
    with no buffering, parallelization, and only splitting on whitespace.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tokenized_count: Counter[Tokens] = Counter()

    def process(self, segment: bytes):
        pre_tokenized = segment.split()
        logger.debug("Processed segment into %d token sequences", len(pre_tokenized))
        tokenized = (tuple(word[i : i + 1] for i in range(len(word))) for word in pre_tokenized)
        tokenized_counts = Counter(tokenized)
        logger.debug("Segment pretokenized into %d unique sequences", len(tokenized_counts))
        self._tokenized_count.update(tokenized_counts)

    def finalize(self) -> Counter[Tokens]:
        logger.info("Finalized pretokenization into %d unique sequences", len(self._tokenized_count))
        return self._tokenized_count
