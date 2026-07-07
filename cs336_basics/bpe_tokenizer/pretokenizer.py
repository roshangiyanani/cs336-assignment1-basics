import logging
from abc import ABC, abstractmethod
from collections import Counter

import regex

logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]
_WORD_RE = regex.compile(rb"\S+")


class Pretokenizer(ABC):
    """
    Performs pre-tokenization on the text corpus in the given file.
    """

    @abstractmethod
    def process(self, segment: bytes | bytearray | memoryview) -> None:
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

    def process(self, segment: bytes | bytearray | memoryview):
        pre_tokenized = _WORD_RE.finditer(segment)
        tokenized = (tuple(match.group()[i : i + 1] for i in range(len(match.group()))) for match in pre_tokenized)
        self._tokenized_count.update(tokenized)
        logger.debug("Updated tokenization count into %d unique sequences", len(self._tokenized_count))

    def finalize(self) -> Counter[Tokens]:
        logger.info("Finalized pretokenization into %d unique sequences", len(self._tokenized_count))
        return self._tokenized_count
