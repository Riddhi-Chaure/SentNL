"""
SentNL — Centralized Text Preprocessor

Every task and every future addition uses this identical pipeline.
Pipeline stages: lowercase → HTML/URL removal → punctuation removal →
number normalisation → tokenization → stopword removal → lemmatization →
whitespace cleanup.
"""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.common.logger import get_logger

logger = get_logger("preprocessing", "system.log")

# Ensure NLTK data is available
_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def _ensure_nltk_data():
    for pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()


class Preprocessor:
    """Stateless text cleaning pipeline driven by configuration flags."""

    # Pre-compiled regex patterns
    _HTML_RE = re.compile(r"<[^>]+>")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _NUMBER_RE = re.compile(r"\d+")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Preprocessing section of the YAML config.
                    Falls back to sensible defaults if None.
        """
        cfg = config or {}
        self.lowercase = cfg.get("lowercase", True)
        self.remove_html = cfg.get("remove_html", True)
        self.remove_urls = cfg.get("remove_urls", True)
        self.remove_punctuation = cfg.get("remove_punctuation", True)
        self.normalize_numbers = cfg.get("normalize_numbers", True)
        self.remove_stopwords = cfg.get("remove_stopwords", True)
        self.lemmatize = cfg.get("lemmatize", True)
        self.min_token_length = cfg.get("min_token_length", 2)

        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on a single text string.

        Args:
            text: Raw input text.

        Returns:
            Cleaned, normalised text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_html:
            text = self._HTML_RE.sub("", text)

        if self.remove_urls:
            text = self._URL_RE.sub("", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.normalize_numbers:
            text = self._NUMBER_RE.sub("", text)

        # Tokenize
        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        if self.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        # Filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Rejoin and collapse whitespace
        cleaned = " ".join(tokens)
        cleaned = self._WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned text strings.
        """
        return [self.clean(t) for t in texts]
