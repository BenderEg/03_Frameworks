"""Агент из 01_ReAct, перенесённый на LangGraph: те же инструменты, тот же системный промпт, та же модель.

Цикл повторяет agent_loop из 01_ReAct/main.py шаг в шаг: не больше max_steps вызовов модели,
при повторе вызова или на последнем шаге — финальный вызов без инструментов. Все запросы к модели идут
через один httpx.Client со счётчиком: из него берутся токены, цена (usage.cost) и первый запрос
каждой конфигурации для сравнения «байт в байт». Счётчик пишется на вопрос в CSV и целиком в results/calls.csv.

Потолок на деньги — кодом, не промптом: max_steps внутри графа и MAX_CALLS_PER_QUESTION на весь вопрос
(charge() перед каждым вызовом модели; паттерны из patterns.py останавливаются раньше, если следующий шаг не влезет).

    uv run python langgraph_agent.py --limit 3
"""

import os, re, json, time, argparse, logging
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import httpx
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage, convert_to_openai_messages)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from utils.wiki_tools import TOOLS, call_tool  # копия utils/ из 01_ReAct без изменений

import answer_check  # локальная копия с правилом 6, см. «Проверку проверки»

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"  # трейсы и CSV голого цикла и sonnet из 01_ReAct, только первые 50 вопросов

load_dotenv(HERE / ".env")

log = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {"HTTP-Referer": "https://postypashki.ru", "X-Title": "agents-course-seminar03"}
QUIZ_PATH = HERE / "data" / "fresh.jsonl"  # копия 01_ReAct/data/fresh.jsonl
TRACES, RESULTS = HERE / "traces", HERE / "results"

# --- скопировано из 01_ReAct/main.py без изменений ---
MODELS = {"cheap": "openai/gpt-4o-mini", "mid": "anthropic/claude-haiku-4.5", "strong": "anthropic/claude-sonnet-4.6"}
SYSTEM = ("You answer questions about the 2026 FIFA World Cup. "
    "If you need a fact, call the tools instead of guessing. "
    "Mention the 2026 FIFA World Cup in your search queries. "
    "When the answer is ready, write it as the last line in the format FINAL: <answer>. "
    "The FINAL answer must be in English, with names and terms spelled exactly as in the source article; "
    "write numbers as digits; give a short phrase, no explanation. "
    "If you could not find the answer, write FINAL: unknown.")
FINISH = ("Tools are no longer available. Answer from what you already know; "
          "write the last line as FINAL: <answer> in English.")
CONFIGS = {
    "без инструментов": [],
    "поиск": ["web_search"],
    "поиск и чтение страницы": ["web_search", "page_find"],
}
ERROR_PREFIX = "ошибка:"


def is_correct(task: dict, answer: str) -> bool:
    if answer.startswith(ERROR_PREFIX):
        return False
    return answer_check.is_correct(task["answer"], answer)


def final_answer(text: str) -> str:
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()
# --- конец копии ---


# Счётчик: каждый ответ OpenRouter записывается с текущей меткой конфигурации и id вопроса.
CALLS: list[dict[str, Any]] = []
FIRST: dict[str, dict[str, Any]] = {}
FIRST_RAW: dict[str, httpx.Request] = {}  # тот же первый запрос как есть: байты тела и заголовки
TAG: dict[str, Any] = {"now": "без метки", "qid": None}

# Потолок на деньги: не больше MAX_CALLS_PER_QUESTION вызовов модели на вопрос, проверяется кодом перед каждым вызовом.
MAX_CALLS_PER_QUESTION = 40
BUDGET = {"start": 0, "limit": MAX_CALLS_PER_QUESTION}


class CallBudgetExceeded(RuntimeError):
    """Вопрос исчерпал потолок вызовов модели."""


def calls_used() -> int:
    return len(CALLS) - BUDGET["start"]


def calls_left() -> int:
    return BUDGET["limit"] - calls_used()


def charge() -> None:
    """Зовётся перед каждым вызовом модели: обрыв кодом, а не просьбой в промпте."""
    if calls_left() <= 0:
        raise CallBudgetExceeded(f"потолок {BUDGET['limit']} вызовов на вопрос исчерпан")


def on_request(request: httpx.Request) -> None:
    """Первый запрос каждой метки целиком — для сравнения с голым циклом байт в байт."""
    if TAG["now"] not in FIRST and request.url.path.endswith("/chat/completions"):
        FIRST[TAG["now"]] = json.loads(request.content)
        FIRST_RAW[TAG["now"]] = request


def on_response(response: httpx.Response) -> None:
    if not response.request.url.path.endswith("/chat/completions"):
        return
    response.read()
    try:
        data = response.json()
    except ValueError:
        log.warning("не-JSON ответ %s: %s", response.status_code, response.text[:200])
        return
    usage = data.get("usage") or {}
    CALLS.append({"tag": TAG["now"], "qid": TAG["qid"], "model": str(data.get("model", "?")).split("/")[-1],
                  "status": response.status_code, "prompt": usage.get("prompt_tokens", 0),
                  "completion": usage.get("completion_tokens", 0), "cost": usage.get("cost") or 0.0,
                  "seconds": round(response.elapsed.total_seconds(), 2)})


HTTP = httpx.Client(event_hooks={"request": [on_request], "response": [on_response]}, timeout=120)


@lru_cache
def chat_model(model: str, temperature: float | None = None) -> ChatOpenAI:
    """По умолчанию без temperature, с usage.include — как тело запроса в chat() из 01_ReAct."""
    return ChatOpenAI(model=model, base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY", ""),
                      http_client=HTTP, default_headers=HEADERS, max_retries=6, temperature=temperature,
                      extra_body={"usage": {"include": True}})


def llm_with_tools(model: str, tool_names: tuple[str, ...] = tuple(CONFIGS["поиск и чтение страницы"]),
                   temperature: float | None = None):
    """Схемы инструментов — готовые OpenAI-словари из wiki_tools: описания аргументов не теряются."""
    if not tool_names:
        return chat_model(model, temperature)
    return chat_model(model, temperature).bind_tools([TOOLS[n]["schema"] for n in tool_names], tool_choice="auto")


class AgentState(MessagesState):
    step: int
    seen: list[str]
    route: str


def call_key(name: str, args: Any) -> str:
    return name + (args if isinstance(args, str) else json.dumps(args, sort_keys=True, ensure_ascii=False))


def pending_calls(msg: AIMessage) -> list[tuple[str, str, Any]]:
    """(id, имя, аргументы) всех вызовов, включая битые: битые получат ошибку инструмента, как в 01_ReAct."""
    return ([(c["id"], c["name"], c["args"]) for c in msg.tool_calls]
            + [(c["id"], c["name"] or "", c["args"] or "") for c in msg.invalid_tool_calls])


def run_tool(call_id: str, name: str, args: Any) -> ToolMessage:
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    try:
        result = call_tool(name, arguments)
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return ToolMessage(content=str(result)[:2000], tool_call_id=call_id)


@lru_cache
def build_graph(model: str, tool_names: tuple[str, ...] = tuple(CONFIGS["поиск и чтение страницы"]),
                max_steps: int = 8, temperature: float | None = None) -> CompiledStateGraph:
    llm = llm_with_tools(model, tool_names, temperature)

    def model_node(state: AgentState) -> dict[str, Any]:
        step = state["step"] + 1
        charge()
        msg = llm.invoke(state["messages"])
        calls = pending_calls(msg)
        keys = [call_key(name, args) for _, name, args in calls]
        repeated = any(k in state["seen"] for k in keys)
        route = "end" if not calls else "finish" if repeated or step == max_steps else "tools"
        return {"messages": [msg], "step": step, "seen": state["seen"] + keys, "route": route}

    def tools_node(state: AgentState) -> dict[str, Any]:
        return {"messages": [run_tool(*c) for c in pending_calls(state["messages"][-1])]}

    def finish_node(state: AgentState) -> dict[str, Any]:
        """Последний ответ с вызовами выбрасывается, модель отвечает без инструментов."""
        last, note = state["messages"][-1], HumanMessage(FINISH)
        charge()
        msg = chat_model(model, temperature).invoke(state["messages"][:-1] + [note])
        return {"messages": [RemoveMessage(id=last.id), note, msg], "step": state["step"] + 1}

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", lambda s: s["route"], {"tools": "tools", "finish": "finish", "end": END})
    graph.add_edge("tools", "model")
    graph.add_edge("finish", END)
    return graph.compile()


@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0


def agent(question: str, model: str, tool_names: list[str] = CONFIGS["поиск и чтение страницы"],
          max_steps: int = 8, temperature: float | None = None, system: str = SYSTEM) -> Run:
    graph = build_graph(model, tuple(tool_names), max_steps, temperature)
    before, started = len(CALLS), time.perf_counter()
    state = graph.invoke({"messages": [SystemMessage(system), HumanMessage(question)], "step": 0, "seen": [],
                          "route": ""}, {"recursion_limit": 2 * max_steps + 5})
    messages: list[BaseMessage] = state["messages"]
    return Run(question, messages[-1].content or "", state["step"], convert_to_openai_messages(messages),
               sum(c["cost"] for c in CALLS[before:]), time.perf_counter() - started)


def load_tasks(path: Path = QUIZ_PATH, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def solve_with_agent(model: str, tool_names: list[str], max_steps: int = 8) -> Callable[[dict], dict[str, Any]]:
    """Голый граф: ответ, трейс и число шагов одного прогона."""
    def solve(task: dict) -> dict[str, Any]:
        run = agent(task["question"], model, tool_names, max_steps)
        return {"answer": run.answer, "messages": run.messages, "steps": run.steps}
    return solve


def save_calls(config: str) -> None:
    """Все вызовы конфигурации в results/calls.csv: отчёт не зависит от памяти процесса."""
    df = pd.DataFrame([c for c in CALLS if c["tag"] == config])
    path = RESULTS / "calls.csv"
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def run_tasks(tasks: list[dict], solve: Callable[[dict], dict[str, Any]], config: str, model_label: str) -> pd.DataFrame:
    """Как run_tasks в 01_ReAct: те же колонки CSV и тот же формат трейса, плюс счётчик на вопрос.

    solve(task) возвращает answer, messages, steps и любые колонки паттерна — они попадают в CSV и трейс;
    необязательный словарь trace уходит только в трейс (длинные истории прогонов).
    """
    TAG["now"] = config
    folder = TRACES / config / model_label
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        TAG["qid"], BUDGET["start"] = task["id"], len(CALLS)
        started = time.perf_counter()
        try:
            out = solve(task)
        except Exception as e:
            log.exception("вопрос %s упал", task["id"])
            out = {"answer": f"{ERROR_PREFIX} {e}", "messages": [], "steps": 0}
        calls = CALLS[BUDGET["start"]:]
        answer, messages, steps = out.pop("answer"), out.pop("messages"), out.pop("steps")
        trace = out.pop("trace", {})
        cost = sum(c["cost"] for c in calls)
        ok = is_correct(task, final_answer(answer))
        rows.append({"config": config, "model": model_label, "id": task["id"], "source": task["source"],
                     "correct": ok, "steps": steps, "tool_calls": sum(m["role"] == "tool" for m in messages),
                     "calls": len(calls), "prompt_tokens": sum(c["prompt"] for c in calls),
                     "completion_tokens": sum(c["completion"] for c in calls), "cost": cost,
                     "seconds": round(time.perf_counter() - started, 1), "answer": final_answer(answer)[:60],
                     "gold": task["answer"], **out})
        (folder / f"{task['id']}.json").write_text(json.dumps({"task": task, "answer": answer, "correct": ok,
                                                              "cost": cost, **out, **trace, "messages": messages},
                                                             ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
        print(f"  {task['id']}: {'+' if ok else '-'} {final_answer(answer)[:60]!r} | эталон {task['answer']!r}",
              flush=True)
    save_calls(config)
    return pd.DataFrame(rows)


def summary_line(df: pd.DataFrame) -> str:
    n, ok = len(df), int(df["correct"].sum())
    tokens = int((df["prompt_tokens"] + df["completion_tokens"]).sum())
    return (f"верно {ok}/{n}, вызовов на вопрос {df['calls'].sum() / n:.1f}, токенов на вопрос {tokens / n:.0f}, "
            f"${df['cost'].sum():.4f} всего, ${df['cost'].sum() / n:.5f} на вопрос")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="прогнать вопросы из data/fresh.jsonl через агента на LangGraph")
    parser.add_argument("--level", default="cheap", choices=list(MODELS))
    parser.add_argument("--config", default="поиск и чтение страницы", choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--tag", default="LangGraph", help="метка конфигурации в счётчике, трейсах и CSV")
    parser.add_argument("--max-steps", type=int, default=8, help="шагов агента до финального вызова без инструментов")
    args = parser.parse_args()
    assert os.getenv("OPENROUTER_API_KEY"), "нет ключа: положите OPENROUTER_API_KEY в .env"

    tag = f"{args.tag}, {args.config}"
    model = MODELS[args.level]
    df = run_tasks(load_tasks(limit=args.limit), solve_with_agent(model, CONFIGS[args.config], args.max_steps), tag,
                   model.split("/")[-1])
    df.to_csv(RESULTS / f"{tag}__{args.level}.csv", index=False, encoding="utf-8")
    print(summary_line(df))
