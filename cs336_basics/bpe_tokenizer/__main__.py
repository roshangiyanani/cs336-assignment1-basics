import argparse
import logging
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm, tqdm_logging_redirect

from cs336_basics.bpe_tokenizer.naive_tokenizer import NaiveTokenizer
from cs336_basics.bpe_tokenizer.pretokenizer import GPT_RE, SimplePretokenizer
from cs336_basics.bpe_tokenizer.segmenter import BufferingSegmenter, InMemorySegmenter

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    logging.basicConfig(format="%(name)s - %(message)s")
    logger.level = logging.INFO

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file.")
    parser.add_argument("filepath", help="Path to the input text file")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Desired vocab size.")
    args = parser.parse_args()

    # segmenter = InMemorySegmenter(SPECIAL_TOKENS)
    segmenter = BufferingSegmenter(SPECIAL_TOKENS)
    segments = segmenter.run(Path(args.filepath))
    pretokenizer = SimplePretokenizer(re=GPT_RE)
    with logging_redirect_tqdm():
        for segment in tqdm(segments, desc="Pretokenizing"):
            pretokenizer.process(segment)

    pretokenized_counts = pretokenizer.finalize()

    tokenizer = NaiveTokenizer(pretokenized_counts, SPECIAL_TOKENS)
    logger.info("Training: %d vocab size", args.vocab_size)
    vocab_size = len(tokenizer.vocab)
    with tqdm_logging_redirect():
        for i in tqdm(range(args.vocab_size - vocab_size)):
            tokenizer.merge_once()

    vocab, merges = tokenizer.as_output()

    logger.info("Training complete. Vocab size: %d, Merges: %d", len(vocab), len(merges))
    logger.debug("Merges: %s", merges)


if __name__ == "__main__":
    main()
