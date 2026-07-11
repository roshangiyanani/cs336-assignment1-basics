import logging
from abc import ABC, abstractmethod
from collections import Counter

import regex
from regex import Pattern

logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]

WORD_RE = regex.compile(rb"\S+")
"""
A regex that performs pre-tokenization as described in the BPE training example, only splitting on whitespace.
"""

GPT_RE = regex.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
"""
This is a slightly prettier form of the original regex used in GPT-2.

Fetched from:
github.com/openai/tiktoken/pull/234/files:
"""


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


class SimplePretokenizer(Pretokenizer):
    """
    Performs simple pre-tokenization, with no buffering and parallelization.
    """

    def __init__(self, re: Pattern[bytes]) -> None:
        super().__init__()
        self.re = re
        self._tokenized_count: Counter[Tokens] = Counter()

    def process(self, segment: bytes | bytearray | memoryview):
        pre_tokenized = self.re.finditer(segment)
        tokenized = (tuple(match.group()[i : i + 1] for i in range(len(match.group()))) for match in pre_tokenized)
        self._tokenized_count.update(tokenized)
        logger.debug("Updated tokenization count into %d unique sequences", len(self._tokenized_count))

    def finalize(self) -> Counter[Tokens]:
        logger.info("Finalized pretokenization into %d unique sequences", len(self._tokenized_count))
        return self._tokenized_count
