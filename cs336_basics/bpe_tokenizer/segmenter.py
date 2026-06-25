import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class Segmenter(ABC):
    """
    Splits an input file into its individual corpus entry segments.
    """

    @abstractmethod
    def run(self, input_path: Path) -> Iterator[bytes]:
        pass


class InMemorySegmenter(Segmenter):
    """
    Loads the entire file into memory before segmenting.
    """

    def __init__(self, special_tokens: Sequence[bytes]):
        super().__init__()
        self._special_tokens_re = re.compile(b"|".join(map(re.escape, special_tokens)))

    def run(self, input_path: Path) -> Iterator[bytes]:
        raw = Path(input_path).read_bytes()
        logger.info("Read %d bytes", len(raw))

        segments_found = 0
        pos = 0
        while (match := self._special_tokens_re.search(raw, pos)) is not None:
            segments_found += 1
            yield raw[pos : match.start()]
            pos = match.end()

        remainder = raw[pos:]
        if remainder:
            segments_found += 1
            yield remainder

        logger.info("Segmented into %d parts.", segments_found)
