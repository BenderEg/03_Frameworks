"""Голый цикл из 01_ReAct, прогнанный через тот же счётчик, что и LangGraph.

В 01_ReAct строка gpt-4o-mini собрана из двух прогонов: провалы получили повторный запуск, удачи не перепроверялись.
Здесь тот же agent_loop — тот же промпт, те же схемы, тот же max_steps и финальный вызов без инструментов — идёт
одним прогоном по всем вопросам. Тело запроса собирает requests, как в chat() из 01_ReAct, и уходит байт в байт
через langgraph_agent.HTTP: токены и цена берутся тем же крючком, а charge() стоит перед каждым вызовом модели.

    uv run python bare_loop.py --limit 3
"""

import os, json, time, argparse, logging, contextlib
from typing import Any

import httpx
import requests

from langgraph_agent import BASE_URL, CONFIGS, HTTP, MODELS, RESULTS, SYSTEM, charge, load_tasks, run_tasks, summary_line
from utils.wiki_tools import TOOLS, call_tool  # копия utils/ из 01_ReAct без изменений

log = logging.getLogger(__name__)

# --- скопировано из 01_ReAct/main.py: заголовки, повторы при 429/5xx; requests.Response заменён на httpx.Response ---
TITLE = {"HTTP-Referer": "https://postypashki.ru", "X-Title": "agents-course-seminar01"}
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_WAIT = 65.0  # окно лимита OpenRouter — минута, плюс запас
FINISH = ("Tools are no longer available. Answer from what you already know; "
          "write the last line as FINAL: <answer> in English.")


def rate_limit_reset(r: httpx.Response) -> float | None:
    """Момент снятия лимита (epoch, секунды): заголовок X-RateLimit-Reset или он же в теле 429 от OpenRouter."""
    reset = r.headers.get("X-RateLimit-Reset")
    if reset is None:
        with contextlib.suppress(ValueError, KeyError, TypeError):
            reset = r.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"]
    with contextlib.suppress(ValueError, TypeError):
        return float(reset) / 1000  # OpenRouter отдаёт миллисекунды
    return None


def retry_delay(r: httpx.Response | None, attempt: int) -> float:
    """Сколько ждать перед повтором: Retry-After, затем X-RateLimit-Reset, иначе экспонента."""
    if r is not None:
        retry_after = r.headers.get("Retry-After")
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                return min(MAX_WAIT, max(1.0, float(retry_after)))
        reset = rate_limit_reset(r)
        if reset is not None:
            return min(MAX_WAIT, max(1.0, reset - time.time() + 1))
    return min(60.0, 3.0 * 2 ** attempt)


def bare_request(body: dict) -> requests.PreparedRequest:
    """Запрос как requests.post(CHAT_URL, json=body, headers=HEADERS) в 01_ReAct: те же байты тела и заголовки."""
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", **TITLE}
    return requests.Session().prepare_request(requests.Request("POST", f"{BASE_URL}/chat/completions",
                                                               json=body, headers=headers))


def post_with_retry(body: dict, attempts: int = 6) -> dict:
    """POST к OpenRouter через счётчик; при 429/5xx и сетевых ошибках ждёт столько, сколько просит сервер."""
    error = "нет попыток"
    prepared = bare_request(body)
    for attempt in range(attempts):
        charge()
        r = None
        try:
            r = HTTP.post(prepared.url, content=prepared.body, headers=dict(prepared.headers))
        except httpx.HTTPError as e:
            label, error = type(e).__name__, f"сеть: {type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            label, error = f"HTTP {r.status_code}", f"HTTP {r.status_code} : {r.text[:5000]}"
            if r.status_code not in RETRY_STATUSES:
                break
        if attempt < attempts - 1:
            delay = retry_delay(r, attempt)
            print(f"  {label}, жду {delay:.1f} c (попытка {attempt + 2}/{attempts})", flush=True)
            time.sleep(delay)
    raise RuntimeError(error)


def chat(messages: list[dict], model: str, tools: list[dict] | None = None) -> dict:
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return post_with_retry(body)["choices"][0]["message"]


def looped(seen: set, calls: list[dict]) -> bool:
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated


def finish(model: str, messages: list[dict], step: int) -> tuple[str, int]:
    del messages[-1]
    messages.append({"role": "user", "content": FINISH})
    msg = chat(messages, model)
    messages.append(msg)
    return msg.get("content") or "", step + 1


def agent_loop(messages: list[dict], model: str, tool_names: list[str], max_steps: int) -> tuple[str, int]:
    seen: set = set()
    for step in range(1, max_steps + 1):
        msg = chat(messages, model, tools=[TOOLS[n]["schema"] for n in tool_names] or None)
        messages.append(msg)
        calls = msg.get("tool_calls") or []

        if not calls:
            return msg.get("content") or "", step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)

        messages += [run_tool(c) for c in calls]
    return "", max_steps  # недостижимо: последний шаг всегда уходит в finish


def run_tool(call: dict) -> dict:
    name = call["function"]["name"]
    try:
        result = call_tool(name, call["function"]["arguments"])
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:2000]}
# --- конец копии ---


def solve_bare(model: str, tool_names: list[str], max_steps: int = 8):
    def solve(task: dict) -> dict[str, Any]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["question"]}]
        answer, steps = agent_loop(messages, model, tool_names, max_steps)
        return {"answer": answer, "messages": messages, "steps": steps}
    return solve


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="голый цикл 01_ReAct одним прогоном через счётчик")
    parser.add_argument("--level", default="cheap", choices=list(MODELS))
    parser.add_argument("--config", default="поиск и чтение страницы", choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--tag", default="голый цикл", help="метка конфигурации в счётчике, трейсах и CSV")
    args = parser.parse_args()
    assert os.getenv("OPENROUTER_API_KEY"), "нет ключа: положите OPENROUTER_API_KEY в .env"

    tag = f"{args.tag}, {args.config}"
    model = MODELS[args.level]
    df = run_tasks(load_tasks(limit=args.limit), solve_bare(model, CONFIGS[args.config]), tag, model.split("/")[-1])
    df.to_csv(RESULTS / f"{tag}__{args.level}.csv", index=False, encoding="utf-8")
    print(summary_line(df))
