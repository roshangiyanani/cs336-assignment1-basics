from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
from cs336_basics.bpe_tokenizer.parallel_seg_and_pretok import ParallelSegmenterandPretokenizer
from cs336_basics.bpe_tokenizer.pretokenizer import GPT_RE, SimplePretokenizer

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PID %(process)d] - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file.")
    parser.add_argument("filepath", help="Path to the input text file")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Desired vocab size.")
    args = parser.parse_args()

    pretokenizer = SimplePretokenizer(re=GPT_RE)
    parallel_processor = ParallelSegmenterandPretokenizer(
        special_tokens=SPECIAL_TOKENS,
        pretokenizer=pretokenizer,
    )

    with tqdm_logging_redirect():
        pretokenized_counts = parallel_processor.process(Path(args.filepath))

        tokenizer = Tokenizer(SPECIAL_TOKENS, pretokenized_counts)
        logger.info("Training: %d vocab size", args.vocab_size)
        vocab_size = len(tokenizer.vocab)

        for i in tqdm(range(args.vocab_size - vocab_size)):
            tokenizer.merge_once()

    vocab, merges = tokenizer.as_output()

    logger.info("Training complete. Vocab size: %d, Merges: %d", len(vocab), len(merges))
    logger.debug("Merges: %s", merges)


if __name__ == "__main__":
    main()
