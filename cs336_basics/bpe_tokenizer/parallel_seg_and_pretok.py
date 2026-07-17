from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Sequence

from tqdm import tqdm

from cs336_basics.bpe_tokenizer.chunker import Chunker
from cs336_basics.bpe_tokenizer.segmenter import BufferingSegmenter
from cs336_basics.bpe_tokenizer.pretokenizer import Pretokenizer, Tokens

logger = logging.getLogger(__name__)

def _process_chunk(
    input_path: Path,
    start: int,
    end: int | None,
    special_tokens: Sequence[str],
    pretokenizer: Pretokenizer,
) -> Counter[Tokens]:
    """
    Worker function to process a single chunk.
    Instantiates a local segmenter and uses the provided pretokenizer.
    """
    logger.debug("Processing chunk [%d:%d] of %s", start, end, input_path)
    segmenter = BufferingSegmenter(special_tokens)

    for segment in segmenter.run(input_path, start=start, end=end):
        pretokenizer.process(segment)

    return pretokenizer.finalize()

class ParallelSegmenterandPretokenizer:
    """
    Parallelly segments and pre-tokenizes an input file.

    This class splits the input file into chunks using `Chunker`,
    processes each chunk in parallel using `ProcessPoolExecutor`,
    and merges the resulting token counts.
    """

    def __init__(
        self,
        special_tokens: Sequence[str],
        pretokenizer: Pretokenizer,
        min_size: int = 2 << 26, # ~64mb
    ) -> None:
        """
        Initializes the parallel segmenter and pretokenizer.

        Args:
            special_tokens: A sequence of special tokens used for segment boundaries.
            pretokenizer: A built Pretokenizer instance to use for tokenization.
            num_workers: The number of worker processes to use. Defaults to
                the number of CPU cores if None.
        """
        self._special_tokens = special_tokens
        self._pretokenizer = pretokenizer
        self._min_size = min_size

    def process(self, input_path: Path, progress: bool = True) -> Counter[Tokens]:
        """
        Parallelly segments and pre-tokenizes the input file.

        Args:
            input_path: The path to the input file.

        Returns:
            A Counter containing the merged token counts.
        """
        chunker = Chunker(special_tokens=self._special_tokens)

        chunks = list(chunker.chunk(input_path, min_size=self._min_size))

        if not chunks:
            logger.info("No chunks found for %s", input_path)
            return Counter()

        total_counts: Counter[Tokens] = Counter()

        logger.info("Starting parallel processing of %d chunks", len(chunks))

        executor = ProcessPoolExecutor()
        try:
            futures = []
            for start, end in chunks:
                futures.append(
                    executor.submit(
                        _process_chunk,
                        input_path,
                        start,
                        end,
                        self._special_tokens,
                        self._pretokenizer,
                    )
                )

            if progress:
                futures = tqdm(as_completed(futures), total=len(futures), desc="Processing chunks")

            for future in futures:
                try:
                    chunk_counts = future.result()
                    total_counts.update(chunk_counts)
                    del chunk_counts
                except Exception as e:
                    logger.error("Worker process failed: %s", e)
                    raise
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received. Cancelling pending tasks...")
            # cancel_futures=True cancels all futures that haven't started yet.
            # wait=False ensures we don't block waiting for already running workers,
            # allowing the program to exit more responsively.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            # If we didn't hit a KeyboardInterrupt, we want a clean, blocking shutdown.
            # If we did hit KeyboardInterrupt, the 'except' block already handled the shutdown.
            # Note: shutdown() is idempotent, so calling it again is safe.
            # If you want to be absolutely sure we don't wait if KeyboardInterrupt happened,
            # you'd need a flag, but shutdown() in finally is generally safe for normal exits.
            # For simplicity and robustness, we call it here.
            # If a KeyboardInterrupt was raised, this might still block if workers are running,
            # but the main thread will proceed to re-raise the KeyboardInterrupt after finally.
            executor.shutdown(wait=True)

        logger.info("Parallel processing completed. Merged %d total counts.", len(total_counts))
        return total_counts
