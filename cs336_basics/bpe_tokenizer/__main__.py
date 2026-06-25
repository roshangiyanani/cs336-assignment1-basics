import argparse
import logging
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cs336_basics.bpe_tokenizer.naive_tokenizer import NaiveTokenizer
from cs336_basics.bpe_tokenizer.pretokenizer import NaivePretokenizer
from cs336_basics.bpe_tokenizer.segmenter import InMemorySegmenter

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = [b"<|endoftext|>"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file.")
    parser.add_argument("filepath", help="Path to the input text file")
    parser.add_argument("--merges", type=int, default=6, help="Number of BPE merges to perform")
    args = parser.parse_args()

    segments = InMemorySegmenter(SPECIAL_TOKENS).run(Path(args.filepath))
    pretokenizer = NaivePretokenizer()
    with logging_redirect_tqdm():
        for segment in tqdm(segments, desc="Pretokenizing"):
            pretokenizer.process(segment)

    pretokenized_counts = pretokenizer.finalize()

    tokenizer = NaiveTokenizer(pretokenized_counts, SPECIAL_TOKENS)
    logger.info("Training: %d merges", args.merges)
    tokenizer.merge_n(args.merges)
    vocab, merges = tokenizer.as_output()

    logger.info("Training complete. Vocab size: %d, Merges: %d", len(vocab), len(merges))
    logger.info("Merges: %s", merges)


if __name__ == "__main__":
    main()
