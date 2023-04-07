from typing import Dict
from typing import List
from typing import Optional


class Codec:
    """
    Respondible for Encoding/Decoding text
    to support tokenization for ChatGPT Modeling

    :param: full_text: Full training data text
    :param: stoi: Map of the full training chars to encoded int
    :param: itos: Map of encoded int to
    """

    def __init__(self, full_text: str):
        chars: List[str] = self._get_unique_chars(full_text)
        self.vocab_size: int = len(chars)
        self.stoi: Dict = self._string_to_int(chars)
        self.itos: Dict = self._int_to_string(chars)

    def _get_unique_chars(self, text: str) -> List[str]:
        """
        Gets set of unique chars from full training data text

        :param: full_text - Full training data text
        :returns: Returns list of unqiue characters
        """
        return sorted(list(set(text)))

    def _string_to_int(self, chars: List[str]) -> Optional[Dict[chr, int]]:
        """
        Maps string to index based on a set of characters

        :param: chars - List of charactors
        :returns: - Dict of index to string
        """
        if isinstance(chars, list):
            return {ch: i for i, ch in enumerate(chars)}

        raise TypeError(f"Must provide List[chars], Recived {type(chars)}")

    def _int_to_string(self, chars: str) -> Optional[Dict[int, chr]]:
        """
        Maps string to index based on a set of characters

        :param: chars - List of charactors
        :returns: - Dict of index to string
        """
        if isinstance(chars, list):
            return {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into list of integers

        :param: text: str - Text provided by client
        :returns: List
        """
        return [self.stoi[c] for c in text]

    def decode(self, encoded: List[int]) -> str:
        """
        Decodes encoded tokenized List

        :param: encoded: tokenized list of strings

        returns: original string
        """
        return ''.join([self.itos[c] for c in encoded])
