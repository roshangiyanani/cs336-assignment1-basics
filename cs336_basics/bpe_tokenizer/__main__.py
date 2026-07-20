from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from cs336_basics.bpe_tokenizer.train import TokenizeTrainer
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

    filepath = Path(args.filepath)

    pretokenizer = SimplePretokenizer(re=GPT_RE)
    parallel_processor = ParallelSegmenterandPretokenizer(
        special_tokens=SPECIAL_TOKENS,
        pretokenizer=pretokenizer,
    )

    with tqdm_logging_redirect():
        pretokenized_counts = parallel_processor.process(filepath)

        tokenizer = TokenizeTrainer(SPECIAL_TOKENS, pretokenized_counts)
        logger.info("Training: %d vocab size", args.vocab_size)
        vocab_size = len(tokenizer.vocab)

        for i in tqdm(range(args.vocab_size - vocab_size)):
            tokenizer.merge_once()

    vocab, merges = tokenizer.as_output()

    logger.info("Training complete. Vocab size: %d, Merges: %d", len(vocab), len(merges))

    longest_token = max(tokenizer.vocab, key=len)
    logger.info("Longest vocab token (%d bytes): %r", len(longest_token), longest_token)

    write_path = filepath.with_suffix("")
    write_path.mkdir(parents=True, exist_ok=True)
    vocab_path = write_path / "vocab.json"
    # Convert bytes to latin-1 strings for JSON serialization and readable text export
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    logger.info("Vocab saved to %s", vocab_path)

    merges_path = write_path / Path("merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for src, tgt in merges:
            f.write(f'"{src.decode("latin-1")}" "{tgt.decode("latin-1")}"\n')
    logger.info("Merges saved to %s", merges_path)

    logger.debug("Merges: %s", merges)


if __name__ == "__main__":
    main()
