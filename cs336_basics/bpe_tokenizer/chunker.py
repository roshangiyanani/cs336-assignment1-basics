from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO
import os
import logging

import regex

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, special_tokens: Sequence[str]):
        if not special_tokens:
            raise ValueError("must pass in at least one special_token for segmentation")

        special_tokens = sorted(special_tokens, key=len, reverse=True)
        for i, token in enumerate(special_tokens):
            if len(token) == 0:
                raise ValueError("special_token must have len > 0")

            if not token.isascii():
                raise ValueError("special_token must be ascii")

            for j in range(i):
                if token in special_tokens[j]:
                    # or else we need to rethink _find_next_boundary
                    raise ValueError("tokens must be non-exclusive")


        super().__init__()

        pat = b"|".join(map(regex.escape, map(str.encode, special_tokens)))
        self._special_tokens_re = regex.compile(pat)

    _BUFFER_SIZE = 1 << 14  # 16kb

    def chunk(self, input: Path, num_chunks: int, min_size: int = 0) -> Iterable[tuple[int, int | None]]:
        """
        Splits a file into roughly equal-sized chunks, aligning boundaries
        to segment boundaries (the position after a special token).

        Returns an iterable of (start, end) byte offset pairs. `end` is `None` for
        the last chunk, meaning \"through the end of the file.\"

        Guaranteed to return <= num_chunks. (TODO: Validate)
        """
        if num_chunks < 1:
            raise ValueError("must have at least 1 chunk")

        if min_size < 0:
            raise ValueError("min_size, if given, must be at least 1")

        with input.open("rb") as f:
            size = f.seek(0, os.SEEK_END)

            if size == 0:
                return []

            chunk_size = (size + num_chunks - 1) // num_chunks
            if chunk_size < min_size:
                logger.warning("chunk_size %d is less than min_size %d. Adjusting to %d", chunk_size, min_size, min_size)
                chunk_size = min_size

            chunk_start = 0
            while chunk_start < size:
                chunk_end = chunk_start + chunk_size
                if chunk_end >= size:
                    logger.info("Last chunk reached the end of file: start=%d, end=None", chunk_start)
                    yield (chunk_start, None)
                    break
                else:
                    f.seek(chunk_end, os.SEEK_SET)
                    next_boundary = self._find_next_boundary(f)
                    if next_boundary is None:
                        logger.info("No more boundaries found. Ending chunks at start=%d, end=None", chunk_start)
                        yield (chunk_start, None)
                        break
                    else:
                        chunk_end += next_boundary
                        logger.debug("Yielding chunk: start=%d, end=%d", chunk_start, chunk_end)
                        yield (chunk_start, chunk_end)

                chunk_start = chunk_end


    def _find_next_boundary(self, f: IO[bytes]) -> int | None:
        """
        Finds the next segment boundary from the given file position.

        A segment boundary is the byte position immediately after a special
        token — i.e., where the next segment begins. Uses buffered reads
        with carry-over so tokens spanning read boundaries are detected.

        Returns the relative file offset of the next boundary, or None if
        no more special tokens exist.

        Leaves the file cursor at or past the end of the special token.
        """

        carry = b""

        while True:
            buffer = f.read(self._BUFFER_SIZE)
            if not buffer:
                return None

            # todo: we don't need to carry the whole string forward
            search = carry + buffer

            if (match := self._special_tokens_re.search(search)) is not None:
                return match.end()

            carry = search

if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the Chunker on a file.")
    parser.add_argument("filepath", type=Path, help="Path to the input file")
    parser.add_argument("--num-chunks", type=int, default=10, help="Number of chunks to split the file into.")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum size of each chunk in bytes.")
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"], help="Special tokens to use for segment boundaries.")
    args = parser.parse_args()

    chunker = Chunker(special_tokens=args.special_tokens)

    chunks = list(chunker.chunk(args.filepath, args.num_chunks, args.min_size))


    print("\n--- Summary ---")
    print(f"Total chunks:\t{len(chunks)}")

    file_size = args.filepath.stat().st_size
    print(f"Total file size:\t{file_size} bytes")

    expected_avg_size = file_size / args.num_chunks
    print(f"Expected chunk size:\t{expected_avg_size}")

    sizes = [(end or file_size) - start for start, end in chunks]
    print(f"Actual chunk sizes:\t{sizes}" )
