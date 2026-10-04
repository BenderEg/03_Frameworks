import re

import httpx

WORLD_CUP_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
HEADERS = {"User-Agent": "agents-course-seminar01/1.0 (https://postypashki.ru; educational project)"}

# Any link back into a 2026_FIFA_World_Cup_* subpage with a fragment identifier.
LINK_RE = re.compile(r'href="(?:https?://en\.wikipedia\.org)?(/wiki/2026_FIFA_World_Cup_\w+#\w+)"')
# Fragments that identify a single match without naming both teams in the URL
# (the final and the third-place match keep a fixed section title instead).
SPECIAL_FRAGMENTS = {"Final", "Match_for_third_place"}


async def fetch_html(url: str = WORLD_CUP_URL) -> str:
    async with httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def extract_match_links(html: str, base: str = "https://en.wikipedia.org") -> list[str]:
    """Extract match-detail links (group and knockout stage), deduped, as absolute URLs."""
    links: list[str] = []
    seen: set[str] = set()
    for path in LINK_RE.findall(html):
        fragment = path.split("#", 1)[1]
        if "_vs_" not in fragment and fragment not in SPECIAL_FRAGMENTS:
            continue
        if path in seen:
            continue
        seen.add(path)
        links.append(base + path)
    return links
