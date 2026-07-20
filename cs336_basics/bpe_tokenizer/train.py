from collections import Counter, defaultdict
from itertools import pairwise
import logging
from collections.abc import Iterable, Sequence
from typing import NamedTuple
import heapq


logger = logging.getLogger(__name__)

Tokens = tuple[bytes, ...]
TokenPair = tuple[bytes, bytes]
Vocabulary = dict[int, bytes]

class _MergeResult(NamedTuple):
    new_word: Tokens
    incremented_pairs: list[TokenPair]
    decremented_pairs: list[TokenPair]

class TokenizeTrainer:
    vocab: list[bytes]
    merges: list[TokenPair]

    def __init__(self, special_tokens: Sequence[str], pretokenized: Counter[Tokens]):
        self.vocab = self._initialize_vocab(special_tokens)
        self.merges = list()
        self.pretokenized_counts = list(pretokenized.items())
        self.token_pair_count_hashmap, self.token_pair_index = self._count_token_pairs(self.pretokenized_counts)
        self.token_pair_count_max_heap = [(count, token_pair) for token_pair, count in self.token_pair_count_hashmap.items()]
        heapq.heapify_max(self.token_pair_count_max_heap)

        logger.info(
            "Tokenizer initialized: %d pretokenized entries, %d token pairs",
            len(self.pretokenized_counts),
            len(self.token_pair_count_max_heap),
        )

    @staticmethod
    def _count_token_pairs(pretokenized_count: Iterable[tuple[Tokens, int]]) -> tuple[Counter[TokenPair], dict[TokenPair, Counter[int]]]:
        """
        Generates the counts for each token-pair, given the pretokenized vocabulary and their counts.
        """
        counts: Counter[tuple[bytes, bytes]] = Counter()
        index: dict[TokenPair, Counter[int]]  = defaultdict(Counter)
        for i, (word, count) in enumerate(pretokenized_count):
            for a, b in pairwise(word):
                token_pair = (a, b)
                counts[token_pair] += count
                index[token_pair][i] += 1

        return counts, index

    @staticmethod
    def _initialize_vocab(special_tokens: Sequence[str]) -> list[bytes]:
        """
        Initialize the vocab using all 256 one-byte sequences and the given special tokens.
        Returns the vocabulary list, where the index is the token ID.
        """
        vocab = [bytes([i]) for i in range(256)]
        vocab.extend(bytes(st, "utf-8") for st in special_tokens)
        logger.info("Vocabulary initialized: %d tokens", len(vocab))
        return vocab

    @staticmethod
    def _indexes_to_merge(pretokenized: Tokens, merge: TokenPair) -> list[int]:
        indexes = []

        i = 0
        max_i = len(pretokenized) - 1
        while i < max_i:
            if (pretokenized[i], pretokenized[i + 1]) == merge:
                indexes.append(i)
                i += 2
            else:
                i += 1

        return indexes

    @staticmethod
    def _apply_merge_and_get_count_diff(word: Tokens, merge_indexes: list[int], replacement: bytes) -> _MergeResult:
        if not merge_indexes:
            return _MergeResult(word, [], [])

        updated_word: list[bytes] = []
        inc_pairs: list[TokenPair] = []
        dec_pairs: list[TokenPair] = []

        prev = 0
        for m in merge_indexes:
            if m != 0:
                before = word[m - 1]
                dec_pairs.append((before, word[m]))
                inc_pairs.append((before, replacement))

            if m + 2 < len(word):
                after = word[m + 2]
                dec_pairs.append((word[m + 1], after))
                inc_pairs.append((replacement, after))

            updated_word.extend(word[prev:m])
            updated_word.append(replacement)
            prev = m + 2

        updated_word.extend(word[prev:])

        return _MergeResult(tuple(updated_word), inc_pairs, dec_pairs)

    def _merge_one(self) -> TokenPair:
        """
        Completes one token merge, updating any index structures as needed, and return the token we merged.
        """
        highest_count, most_common_pair = heapq.heappop_max(self.token_pair_count_max_heap)
        while self.token_pair_count_hashmap[most_common_pair] != highest_count:
            highest_count, most_common_pair = heapq.heappop_max(self.token_pair_count_max_heap)

        token = b"".join(most_common_pair)
        logger.debug("Merge: %r + %r → %r (count: %d)", most_common_pair[0], most_common_pair[1], token, highest_count)

        token_pairs_delta: dict[TokenPair, int] = defaultdict(int) # int() == 0

        for i, count in self.token_pair_index[most_common_pair].items():
            if count > 0:
                word, count = self.pretokenized_counts[i]
                indexes = self._indexes_to_merge(word, most_common_pair)
                self.token_pair_index[most_common_pair][i] -= len(indexes)

                new_word, inc_pairs, dec_pairs = self._apply_merge_and_get_count_diff(word, indexes, token)
                for inc_pair in inc_pairs:
                    self.token_pair_index[inc_pair][i] += 1
                    token_pairs_delta[inc_pair] += count

                for dec_pair in dec_pairs:
                    self.token_pair_index[dec_pair][i] -= 1
                    token_pairs_delta[dec_pair] -= count

                self.pretokenized_counts[i] = new_word, count

        del self.token_pair_count_hashmap[most_common_pair]
        for token_pair, delta in token_pairs_delta.items():
            self.token_pair_count_hashmap[token_pair] += delta
            heapq.heappush_max(self.token_pair_count_max_heap, (self.token_pair_count_hashmap[token_pair], token_pair))

        return most_common_pair

    def merge_once(self) -> None:
        most_common_pair = self._merge_one()
        self.merges.append(most_common_pair)
        self.vocab.append(b"".join(most_common_pair))

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
