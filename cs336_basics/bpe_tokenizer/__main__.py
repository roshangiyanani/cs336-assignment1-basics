import argparse
import logging

from cs336_basics.bpe_tokenizer.naive_tokenizer import NaiveTokenizer
from cs336_basics.bpe_tokenizer.pretokenizer import NaivePretokenizer

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file.")
    parser.add_argument("filepath", help="Path to the input text file")
    args = parser.parse_args()

    logger.info("Pretokenizing file: %s", args.filepath)
    special_tokens = [""]
    pretokenized = NaivePretokenizer().run(args.filepath, special_tokens)
    logger.info("Pretokenization complete: %d unique token sequences", len(pretokenized))

    logger.info("Initializing tokenizer")
    tokenizer = NaiveTokenizer(pretokenized, special_tokens)
    logger.info("Training: %d merges", 6)
    tokenizer.merge_n(6)
    vocab, merges = tokenizer.as_output()

    logger.info("Training complete. Vocab size: %d, Merges: %d", len(vocab), len(merges))
    logger.info("Merges: %s", merges)


if __name__ == "__main__":
    main()
