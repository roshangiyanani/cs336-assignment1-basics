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
    def run(self, input_path: Path, start: int = 0, end: int | None = None) -> Iterator[str]:
        """
        Splits an input file into its individual corpus entry segments.

        Args:
            input_path: The path to the input file.
            start: The starting byte position (inclusive). Defaults to 0.
            end: The ending byte position (exclusive). Defaults to None (end of file).
        """
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

            if not token.isascii():
                raise ValueError("special_token must be ascii")

        super().__init__()

        pat = b"|".join(map(regex.escape, map(str.encode, special_tokens)))
        self._special_tokens_re = regex.compile(pat)

    def run(self, input_path: Path, start: int = 0, end: int | None = None) -> Iterator[str]:
        """
        Splits an input file into its individual corpus entry segments.

        Args:
            input_path: The path to the input file.
            start: The starting byte position (inclusive). Defaults to 0.
            end: The ending byte position (exclusive). Defaults to None (end of file).
        """
        requires_seeking = start != 0 or end is not None
        raw = input_path.read_bytes()
        if requires_seeking:
            raw = raw[start:end]
        logger.info("Read %d bytes", len(raw))

        segments_found = 0
        pos = 0
        while (match := self._special_tokens_re.search(raw, pos)) is not None:
            segments_found += 1
            yield raw[pos : match.start()].decode("utf-8")
            pos = match.end()

        remainder = raw[pos:]
        if remainder:
            segments_found += 1
            yield remainder.decode("utf-8")

        logger.info("Segmented into %d parts.", segments_found)


class BufferingSegmenter(Segmenter):
    """
    Segments the file during stream reading, returning each segment.
    """

    _DEFAULT_BUFFER_SIZE = 1 << 16  # 64kb

    def __init__(self, special_tokens: Sequence[str], buffer_size: int = _DEFAULT_BUFFER_SIZE):
        if not special_tokens:
            raise ValueError("must pass in at least one special_token for segmentation")

        for token in special_tokens:
            if len(token) == 0:
                raise ValueError("special_token must have len > 0")

            if not token.isascii():
                raise ValueError("special_token must be ascii")

        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        super().__init__()
        pat = b"|".join(map(regex.escape, map(str.encode, special_tokens)))
        self._special_tokens_re = regex.compile(pat)
        self._buffer_size = buffer_size

    def run(self, input_path: Path, start: int = 0, end: int | None = None) -> Iterator[str]:
        """
        Splits an input file into its individual corpus entry segments.

        Args:
            input_path: The path to the input file.
            start: The starting byte position (inclusive). Defaults to 0.
            end: The ending byte position (exclusive). Defaults to None (end of file).
        """
        logger.info("Starting to segment %s", input_path)
        with input_path.open("rb") as f:
            if start != 0:
                logger.debug("seeking to %d", start)
                f.seek(start)

            carry = b""
            reads_issued = 0
            pos = start
            segments_found = 0

            while end is None or pos < end:
                reads_issued += 1
                to_read = self._buffer_size if end is None else min(self._buffer_size, end - pos)
                buffer = f.read(to_read)
                pos += len(buffer)
                logger.debug("Read %d bytes in read %d", len(buffer), reads_issued)
                if len(buffer) == 0:
                    break

                search = carry + buffer
                search_pos = 0
                # todo: don't need to start at pos, can start at like `len(carry) - self._carry_over_len`?
                while (match := self._special_tokens_re.search(search, search_pos)) is not None:
                    segments_found += 1
                    yield search[search_pos : match.start()].decode()
                    search_pos = match.end()

                if search_pos == 0:
                    carry = search
                elif search_pos == len(search):
                    carry = b""
                else:
                    carry = search[search_pos:]

        if carry:
            segments_found += 1
            yield carry.decode()
        logger.info("Segmented %d bytes into %d parts from %d reads.", pos-start, segments_found, reads_issued)
