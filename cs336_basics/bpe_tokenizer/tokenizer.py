import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]
TokenPair = tuple[bytes, bytes]
Vocabulary = dict[int, bytes]


class Tokenizer(ABC):
    vocab: list[bytes]
    merges: list[TokenPair]

    def __init__(self, special_tokens: Sequence[str]):
        self.vocab = self._initialize_vocab(special_tokens)
        self.merges = list()

    @staticmethod
    def _initialize_vocab(special_tokens: Sequence[str]) -> list[bytes]:
        """
        Initialize the vocab using all 256 one-byte sequences and the given special tokens.
        Returns the vocabulary list, where the index is the token ID.
        """
        vocab = [chr(i).encode("utf-8") for i in range(256)]
        vocab += [special_token.encode("utf-8") for special_token in special_tokens]
        logger.info("Vocabulary initialized: %d tokens", len(vocab))
        return vocab

    @abstractmethod
    def merge_once(self):
        pass

    def merge_n(self, count: int) -> None:
        logger.info("Running %d merges", count)
        for step in range(count):
            self.merge_once()
        logger.info("Completed %d merges. Vocab size: %d", count, len(self.vocab))

    def merge_until(self, vocab_size: int) -> None:
        if len(self.vocab) > vocab_size:
            raise ValueError("vocab_size is already too large")

        self.merge_n(vocab_size - len(self.vocab))

    def as_output(self) -> tuple[Vocabulary, list[TokenPair]]:
        return dict(enumerate(self.vocab)), self.merges
