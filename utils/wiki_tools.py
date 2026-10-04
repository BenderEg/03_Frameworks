"""Инструменты агента поверх Wikipedia API: web_search и page_find.

Общие для main.py и lesson/fetch_fresh.py. Единственная точка выхода в сеть — wiki(),
её и подменяют в тестах.
"""

import re
import time
from typing import Any, Callable

import requests
from pydantic import BaseModel, Field

from utils.wiki_links import HEADERS as WIKI_HEADERS
from utils.wiki_text import html_lines

WIKI = "https://en.wikipedia.org/w/api.php"


def wiki(params: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    """Ответ Википедии для params["action"] (query или parse); {} если страницы нет."""
    problem = "нет ответа"
    for attempt in range(attempts):
        try:
            r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=20)
        except requests.RequestException as e:
            problem = type(e).__name__
        else:
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
                if "error" in data:
                    if data["error"].get("code") == "missingtitle":
                        return {}
                    raise RuntimeError(f"Википедия: {data['error'].get('info')}")
                return data[params["action"]]
            problem = f"HTTP {r.status_code}"
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия недоступна: {problem}")


def page_html(title: str, intro: bool = False) -> str:
    """HTML статьи после парсера, с шаблонами (footballbox, инфобоксы); intro=True — только вводная секция."""
    params: dict[str, Any] = {"action": "parse", "page": title, "prop": "text", "redirects": 1, "formatversion": 2}
    if intro:
        params["section"] = 0
    return wiki(params).get("text", "")


class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")


class PageFindArgs(BaseModel):
    title: str = Field(description="точное название статьи")
    keywords: str = Field(description="от двух до пяти ключевых слов из вопроса")


def web_search(query: str) -> str:
    """Интро лучшей статьи по запросу и заголовки ещё двух."""
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["search"]
    if not hits:
        return "nothing found"
    text = " ".join(html_lines(page_html(hits[0]["title"], intro=True)))[:1500]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")


def page_find(title: str, keywords: str) -> str:
    """Строки статьи (предложения, строки таблиц, карточки матчей footballbox), где встречается
    хотя бы половина ключевых слов; сначала строки с наибольшим числом совпадений."""
    html = page_html(title)
    if not html:
        return "no such page"
    words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    need = max(1, (len(words) + 1) // 2)
    scored = [(sum(w in l.lower() for w in words), l) for l in html_lines(html)]
    hits = [l for s, l in sorted(scored, key=lambda p: -p[0]) if s >= need]
    return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"


ToolSpec = dict[str, Any]

TOOLS: dict[str, ToolSpec] = {
    "web_search": {"fn": web_search, "args": SearchArgs, "schema": {"type": "function", "function": {
        "name": "web_search", "description": "Searches English Wikipedia and returns the intro of the best article and the titles of two more",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "short query: a name, a title, a term"}},
                       "required": ["query"]}}}},
    "page_find": {"fn": page_find, "args": PageFindArgs, "schema": {"type": "function", "function": {
        "name": "page_find", "description": "Looks inside the full text of a Wikipedia article (for example '2026 in Japan') and returns the lines containing the keywords",
        "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "exact article title"},
                                                        "keywords": {"type": "string", "description": "two to five keywords from the question"}},
                       "required": ["title", "keywords"]}}}},
}


def call_tool(name: str, arguments: str) -> str:
    """Вызвать инструмент по имени и JSON-аргументам от модели; ошибки валидации и сети пробрасываются."""
    spec = TOOLS[name]
    fn: Callable[..., str] = spec["fn"]
    return fn(**spec["args"].model_validate_json(arguments).model_dump())
