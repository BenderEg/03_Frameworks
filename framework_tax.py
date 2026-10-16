"""Налог фреймворка на пустом запросе: первый вызов модели у голого цикла 01_ReAct и у LangGraph на простом вопросе.

Оба запроса идут через один httpx.Client со счётчиком (langgraph_agent.HTTP): токены берутся из usage ответа
OpenRouter, а тела и заголовки сравниваются байт в байт и как JSON. Голый цикл слал запрос через requests, поэтому
его тело и заголовки собираются тем же requests (prepare_request) и отправляются как есть.

    uv run python framework_tax.py
"""

import os, json, difflib

from bare_loop import bare_request
from langgraph_agent import CALLS, CONFIGS, FIRST_RAW, MODELS, RESULTS, SYSTEM, TAG, TOOLS, HTTP, agent

QUESTION = "Which country will host the 2026 FIFA World Cup final?"
BARE, FRAMEWORK = "налог: голый цикл", "налог: LangGraph"
OUT = RESULTS / "first_request"
# Ключ не пишем; длина и хост различаются по построению; cookie — это __cf_bm Cloudflare из ответа на предыдущий
# запрос, общий httpx.Client хранит его и шлёт со следующим.
HIDDEN = {"authorization", "content-length", "host", "cookie"}


def first_body(question: str, model: str) -> dict:
    """Тело первого вызова agent_loop из 01_ReAct: как в chat() с инструментами конфигурации."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    tools = [TOOLS[n]["schema"] for n in CONFIGS["поиск и чтение страницы"]]
    return {"model": model, "messages": messages, "usage": {"include": True}, "tools": tools, "tool_choice": "auto"}


def first_call(tag: str) -> dict:
    return next(c for c in CALLS if c["tag"] == tag)


def shown_headers(headers) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower() not in HIDDEN}


def main() -> None:
    model = MODELS["cheap"]
    TAG["now"] = BARE
    prepared = bare_request(first_body(QUESTION, model))
    HTTP.post(prepared.url, content=prepared.body, headers=dict(prepared.headers)).raise_for_status()
    bare_raw = FIRST_RAW[BARE].content

    TAG["now"] = FRAMEWORK
    run = agent(QUESTION, model)
    graph_raw = FIRST_RAW[FRAMEWORK].content

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bare.json").write_bytes(bare_raw)
    (OUT / "langgraph.json").write_bytes(graph_raw)
    bare_headers, graph_headers = shown_headers(prepared.headers), shown_headers(FIRST_RAW[FRAMEWORK].headers)
    (OUT / "headers.json").write_text(json.dumps({"голый цикл": bare_headers, "LangGraph": graph_headers},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")

    bare_body, graph_body = json.loads(bare_raw), json.loads(graph_raw)
    compact = len(json.dumps(bare_body, separators=(",", ":")).encode())
    without_stream = {k: v for k, v in graph_body.items() if k != "stream"}
    canonical = [json.dumps(b, sort_keys=True, ensure_ascii=False, indent=1).splitlines()
                 for b in (bare_body, graph_body)]
    body_diff = list(difflib.unified_diff(*canonical, "голый цикл", "LangGraph", lineterm="", n=1))
    header_diff = {k: (bare_headers.get(k, "—"), graph_headers.get(k, "—"))
                   for k in sorted(bare_headers.keys() | graph_headers.keys())
                   if bare_headers.get(k) != graph_headers.get(k)}
    tokens = {name: first_call(tag)["prompt"] for name, tag in (("голый цикл", BARE), ("LangGraph", FRAMEWORK))}

    lines = [
        "# Налог фреймворка на пустом запросе", "", f"Вопрос: «{QUESTION}», модель {model}, первый вызов.", "",
        "| | голый цикл | LangGraph |", "| --- | --: | --: |",
        f"| prompt_tokens из usage | {tokens['голый цикл']} | {tokens['LangGraph']} |",
        f"| байт в теле запроса | {len(bare_raw)} | {len(graph_raw)} |",
        f"| ключи тела | {', '.join(bare_body)} | {', '.join(graph_body)} |", "",
        f"Тела совпадают байт в байт: **{bare_raw == graph_raw}**; как JSON: **{bare_body == graph_body}**; "
        f"порядок ключей тот же: **{list(bare_body) == list(graph_body)}**; как JSON без `stream`: "
        f"**{without_stream == bare_body}**.", "",
        f"Байты: requests пишет JSON с пробелами после `,` и `:` — без них тело голого цикла {compact} байт; "
        f"LangGraph компактен и добавляет `\"stream\":false,` ({len(graph_raw) - compact} байт).", "",
        "Разница тел как JSON (ключи отсортированы):", "", "```diff", *(body_diff or ["(нет)"]), "```", "",
        "Разница заголовков (без Authorization, Content-Length, Host и cookie Cloudflare):", "",
        "| заголовок | голый цикл | LangGraph |", "| --- | --- | --- |",
        *[f"| {k} | {a} | {b} |" for k, (a, b) in header_diff.items()], "",
        f"Вызовов модели у LangGraph на этом вопросе: {run.steps}.",
    ]
    report = "\n".join(lines)
    (RESULTS / "framework_tax.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    assert os.getenv("OPENROUTER_API_KEY"), "нет ключа: положите OPENROUTER_API_KEY в .env"
    main()
