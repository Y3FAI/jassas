"""
Parser - HTML to clean text with Arabic normalization.
"""
import re
import unicodedata
from bs4 import BeautifulSoup


class Parser:
    """Parses HTML and extracts normalized text."""

    # Tags to remove completely
    NOISE_TAGS = ['script', 'style', 'nav', 'footer', 'header', 'meta', 'noscript', 'aside', 'iframe']

    def parse(self, html: str) -> dict:
        """
        Parse HTML and return cleaned metadata.

        Args:
            html: Raw HTML content

        Returns:
            dict with keys:
                - title: str (cleaned title)
                - clean_text: str (normalized body text)
                - doc_len: int (word count for BM25)
        """
        if not html or not html.strip():
            return {'title': '', 'clean_text': '', 'doc_len': 0}

        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            return {'title': '', 'clean_text': '', 'doc_len': 0}

        # Remove noise tags
        for tag in soup(self.NOISE_TAGS):
            tag.decompose()

        # Extract title
        title = self._extract_title(soup)

        # Extract body text
        text = soup.get_text(separator=' ')

        # Normalize
        clean_text = self._normalize(text)
        clean_title = self._normalize(title)

        return {
            'title': clean_title,
            'clean_text': clean_text,
            'doc_len': len(clean_text.split())
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from <title> or <h1>."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        if soup.h1 and soup.h1.string:
            return soup.h1.string.strip()
        # Try first h1 with text
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _normalize(self, text: str) -> str:
        """
        Normalize text with Arabic-specific rules.

        Steps:
        1. Unicode NFKC normalization
        2. Remove Arabic diacritics (tashkeel)
        3. Unify Alif variants (أ إ آ → ا)
        4. Unify Teh Marbuta (ة → ه)
        5. Unify Yeh (ى → ي)
        6. Collapse whitespace
        7. Lowercase (for English parts)
        """
        if not text:
            return ""

        # Unicode normalize
        text = unicodedata.normalize('NFKC', text)

        # Remove Arabic diacritics (tashkeel)
        # Range: \u064B-\u065F (fatha, damma, kasra, shadda, sukun, etc.)
        # Plus \u0670 (superscript alef)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

        # Unify Alif variants: أ إ آ → ا
        text = re.sub(r'[أإآ]', 'ا', text)

        # Unify Teh Marbuta: ة → ه
        text = re.sub(r'ة', 'ه', text)

        # Unify Yeh: ى (alef maksura) → ي
        text = re.sub(r'ى', 'ي', text)

        # Strip Arabic definite article (ال) with length guard
        # Only strip from words > 4 chars to protect roots like الله, الا
        # "السيبراني" (9 chars) → "سيبراني" ✓
        # "الله" (4 chars) → "الله" (protected) ✓
        words = text.split()
        words = [re.sub(r'^ال', '', w) if w.startswith('ال') and len(w) > 4 else w for w in words]
        text = ' '.join(words)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Lowercase for English parts
        return text.lower()
