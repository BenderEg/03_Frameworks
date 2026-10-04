import re

from bs4 import BeautifulSoup

# Elements with no facts in them: styles, footnote markers, edit links, navigation, hidden
# microformat duplicates (e.g. ISO dates next to the human-readable one).
JUNK_SELECTOR = (
    "style, script, sup.reference, .mw-editsection, .hatnote, .navbox, .reflist, .mw-references-wrap, "
    'span.bday, [style*="display:none"], [style*="display: none"]'
)
BLOCK_TAGS = ("li", "tr", "dd", "dt", "caption", "h2", "h3", "h4", "div", "table")
# Sentence end: a lowercase letter or digit before the dot, so "U.S. President" stays whole.
SENTENCE_END = re.compile(r"(?<=[a-z0-9][.!?])\s+(?=[A-Z])")


def _tidy(line: str) -> str:
    """Collapse whitespace and drop spaces that get_text(" ") puts around punctuation."""
    line = " ".join(line.split())
    line = re.sub(r"\s+([,.;:)\]])", r"\1", line)
    return re.sub(r"([(\[])\s+", r"\1", line)


def html_lines(html: str) -> list[str]:
    """Turn MediaWiki parser HTML into text lines: one per sentence, list item, table row or footballbox.

    A footballbox (date, teams, score, scorers, venue, attendance, referee) becomes a single line,
    so a keyword search over lines finds the teams and the match facts together.
    """
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select(JUNK_SELECTOR):
        junk.decompose()
    for box in soup.select("div.footballbox"):
        box.replace_with("\n" + _tidy(box.get_text(" ", strip=True)) + "\n")
    for paragraph in soup.find_all("p"):
        sentences = SENTENCE_END.split(_tidy(paragraph.get_text(" ", strip=True)))
        paragraph.replace_with("\n" + "\n".join(sentences) + "\n")
    for block in soup.find_all(BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")
    return [line for line in map(_tidy, soup.get_text(" ").splitlines()) if line]
