import logging
from abc import ABC, abstractmethod
from collections.abc import Buffer, Iterator, Sequence
from pathlib import Path

import regex

logger = logging.getLogger(__name__)


class Segmenter(ABC):
    """
    Splits an input file into its individual corpus entry segments.
    """

    @abstractmethod
    def run(self, input_path: Path) -> Iterator[str]:
        pass


class InMemorySegmenter(Segmenter):
    """
    Loads the entire file into memory before segmenting.
    """

    def __init__(self, special_tokens: Sequence[str]):
        if not special_tokens:
            raise ValueError("must pass in at least one special_token for segmentation")

        for token in special_tokens:
            if len(token) == 0:
                raise ValueError("special_token must have len > 0")

        super().__init__()
        self._special_tokens_re = regex.compile("|".join(map(regex.escape, special_tokens)))

    def run(self, input_path: Path) -> Iterator[str]:
        raw = input_path.read_text()
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


class BufferingSegmenter(Segmenter):
    """
    Segments the file during stream reading, returning each segment.
    """

    _DEFAULT_BUFFER_SIZE = 1 << 16  # 64kb

    def __init__(self, special_tokens: Sequence[str], buffer_size: int = _DEFAULT_BUFFER_SIZE):
        if not special_tokens:
            raise ValueError("must pass in at least one special_token for segmentation")

        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        super().__init__()
        self._special_tokens_re = regex.compile("|".join(map(regex.escape, special_tokens)))
        self._buffer_size = buffer_size

    def run(self, input_path: Path) -> Iterator[str]:
        logger.info("Starting to segment %s", input_path)
        with input_path.open("r") as f:
            carry = ""
            reads_issued = 0
            amt_read = 0
            segments_found = 0

            while True:
                reads_issued += 1
                buffer = f.read(self._buffer_size)
                amt_read += len(buffer)
                logger.debug("Read %d bytes in read %d", len(buffer), reads_issued)
                if len(buffer) == 0:
                    if carry:
                        segments_found += 1
                        yield carry
                    break

                search = carry + buffer
                pos = 0
                # todo: don't need to start at pos, can start at like `len(carry) - self._carry_over_len`?
                while (match := self._special_tokens_re.search(search, pos)) is not None:
                    segments_found += 1
                    yield search[pos : match.start()]
                    pos = match.end()

                if pos == 0:
                    carry = search
                elif pos == len(search):
                    carry = ""
                else:
                    carry = search[pos:]

        logger.info("Segmented %d bytes into %d parts from %d reads.", amt_read, segments_found, reads_issued)
