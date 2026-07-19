import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import NamedTuple

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer, TokenPair, Tokens

logger = logging.getLogger(__name__)


class _MergeResult(NamedTuple):
    new_word: Tokens
    incremented_pairs: list[TokenPair]
    decremented_pairs: list[TokenPair]


class NaiveTokenizer(Tokenizer):
    """
    Implements the basic tokenizer, with very few optimizations.
    """

    def __init__(self, pretokenized: Counter[Tokens], special_tokens: Sequence[str]):
        super().__init__(special_tokens)
        self.pretokenized_counts = list(pretokenized.items())
        self.token_pair_counts, self.token_pair_index = self._count_token_pairs(self.pretokenized_counts)
        logger.info(
            "Tokenizer initialized: %d pretokenized entries, %d token pairs",
            len(self.pretokenized_counts),
            len(self.token_pair_counts),
        )

    def merge_once(self) -> None:
        most_common_pair = self._merge_one()
        self.merges.append(most_common_pair)
        self.vocab.append(b"".join(most_common_pair))

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

    def _merge_one(self) -> TokenPair:
        """
        Completes one token merge, updating any index structures as needed, and return the token we merged.
        """
        _, highest_count = self.token_pair_counts.most_common(1)[0]
        most_common_pair = max(token for token, count in self.token_pair_counts.items() if count == highest_count)
        token = b"".join(most_common_pair)
        logger.debug("Merge: %r + %r → %r (count: %d)", most_common_pair[0], most_common_pair[1], token, highest_count)

        for i, count in self.token_pair_index[most_common_pair].items():
            if count > 0:
                word, count = self.pretokenized_counts[i]
                indexes = NaiveTokenizer._indexes_to_merge(word, most_common_pair)
                self.token_pair_index[most_common_pair][i] -= len(indexes)

                new_word, inc_pairs, dec_pairs = NaiveTokenizer._apply_merge_and_get_count_diff(word, indexes, token)
                for inc_pair in inc_pairs:
                    self.token_pair_index[inc_pair][i] += 1
                    self.token_pair_counts[inc_pair] += count

                for dec_pair in dec_pairs:
                    self.token_pair_index[dec_pair][i] -= 1
                    self.token_pair_counts[dec_pair] -= count

                self.pretokenized_counts[i] = new_word, count

        del self.token_pair_counts[most_common_pair]

        return most_common_pair

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
