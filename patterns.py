"""Два паттерна недели поверх агента на LangGraph: судья с опорой на инструменты и голосование с каскадом.

Судья: агент отвечает, судья сам ищет подтверждение в Википедии теми же инструментами и выносит вердикт
по схеме Verdict; при отказе агент получает замечание и ищет заново — не больше двух попыток исправиться.
Голосование: три независимых прогона gpt-4o-mini при temperature 0.7, ответ большинством (2 из 3, не unknown);
без большинства вопрос уходит тому же агенту на claude-haiku-4.5.
Повтор при unknown — контроль к судье: те же раунды и то же замечание на unknown, но без проверяющего агента.

Оба паттерна — графы LangGraph поверх agent() из langgraph_agent.py, все вызовы идут через тот же счётчик,
а потолок MAX_CALLS_PER_QUESTION проверяется кодом перед каждым вызовом модели.

    uv run python patterns.py --pattern judge --limit 3
    uv run python patterns.py --pattern vote --limit 3
    uv run python patterns.py --pattern retry --limit 3
"""

import os, argparse, logging, operator
from collections import Counter
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from langgraph_agent import (BUDGET, CONFIGS, MODELS, RESULTS, ERROR_PREFIX, agent, answer_check, calls_left,
                             charge, chat_model, final_answer, is_correct, load_tasks, run_tasks, summary_line)

log = logging.getLogger(__name__)

CHEAP, MID = MODELS["cheap"], MODELS["mid"]
TOOL_NAMES = CONFIGS["поиск и чтение страницы"]
MAX_STEPS, JUDGE_STEPS = 8, 4
ANSWER_CALLS = MAX_STEPS + 1          # прогон агента: max_steps шагов и, возможно, финальный вызов без инструментов
JUDGE_CALLS = JUDGE_STEPS + 1 + 1     # прогон судьи и вызов за вердиктом по схеме
MAX_ROUNDS = 3                        # первый ответ и две попытки исправиться
VOTES, VOTE_TEMPERATURE = 3, 0.7

# --- судья ---

JUDGE_SYSTEM = ("You fact-check a proposed answer to a question about the 2026 FIFA World Cup. "
                "Do not trust the proposed answer or your memory: use the tools to find the Wikipedia passage "
                "that confirms or contradicts it. Mention the 2026 FIFA World Cup in your search queries. "
                "Finish with a short report: quote the passage you found and say whether it supports the answer.")
VERDICT_SYSTEM = ("You decide whether a proposed answer is supported. Judge only by the quoted evidence. "
                  "If the evidence does not contain the answer, or the answer is 'unknown', it is not supported. "
                  "If not supported, say in the issue what is wrong and what the evidence says instead.")
RETRY_NOTE = ("\n\nA reviewer rejected your previous answer '{answer}': {issue} "
              "Search again and correct it.")


class Verdict(BaseModel):
    supported: bool = Field(description="true only if the evidence explicitly supports the answer")
    issue: str = Field(description="what is wrong or missing and what the evidence says, empty if supported")


class JudgeState(TypedDict):
    question: str
    answer: str
    first_answer: str
    evidence: str
    feedback: str
    rounds: int
    steps: int
    verdicts: list[dict[str, Any]]
    history: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    budget_hit: bool
    route: str


def is_unknown(answer: str) -> bool:
    return answer_check.answer_tokens(final_answer(answer)) in ([], ["unknown"])


def tool_outputs(messages: list[dict[str, Any]], limit: int) -> str:
    found = "\n".join(str(m["content"])[:800] for m in messages if m["role"] == "tool")
    return found[-limit:]


def answer_node(state: JudgeState) -> dict[str, Any]:
    note = RETRY_NOTE.format(answer=final_answer(state["answer"]), issue=state["feedback"]) if state["feedback"] else ""
    run = agent(state["question"] + note, CHEAP, TOOL_NAMES, MAX_STEPS)
    rounds = state["rounds"] + 1
    route, budget_hit = "judge", False
    if rounds >= MAX_ROUNDS:
        route = "end"
    elif calls_left() < JUDGE_CALLS:
        route, budget_hit = "end", True
    return {"answer": run.answer, "first_answer": state["first_answer"] or run.answer,
            "evidence": tool_outputs(run.messages, 4000), "rounds": rounds, "steps": state["steps"] + run.steps,
            "messages": run.messages, "budget_hit": budget_hit, "route": route,
            "history": state["history"] + [{"node": "answer", "round": rounds, "answer": run.answer,
                                            "messages": run.messages}]}


def judge_node(state: JudgeState) -> dict[str, Any]:
    answer = final_answer(state["answer"])
    claim = (f"Question: {state['question']}\nProposed answer: {answer}\n\n"
             f"Evidence the answering agent retrieved:\n{state['evidence'] or '(nothing)'}")
    run = agent(claim, CHEAP, TOOL_NAMES, JUDGE_STEPS, system=JUDGE_SYSTEM)
    charge()
    verdict: Verdict = chat_model(CHEAP).with_structured_output(Verdict).invoke([
        SystemMessage(VERDICT_SYSTEM),
        HumanMessage(f"Question: {state['question']}\nProposed answer: {answer}\n\n"
                     f"Evidence found by the checker:\n{tool_outputs(run.messages, 4000) or '(nothing)'}\n\n"
                     f"Checker's report:\n{run.answer}")])
    if verdict.supported and is_unknown(answer):
        # v1 судьи принимал unknown в 15 вопросах из 50 вопреки VERDICT_SYSTEM: правило теперь в коде,
        # а находки проверяющего уходят агенту замечанием.
        verdict = Verdict(supported=False, issue=f"'unknown' is not an answer. The checker found: {run.answer[-600:]}")
    route, budget_hit = "answer", False
    if verdict.supported:
        route = "end"
    elif calls_left() < ANSWER_CALLS:
        route, budget_hit = "end", True
    record = {"round": state["rounds"], "answer": answer, **verdict.model_dump()}
    return {"verdicts": state["verdicts"] + [record], "feedback": verdict.issue, "steps": state["steps"] + run.steps + 1,
            "budget_hit": budget_hit, "route": route,
            "history": state["history"] + [{"node": "judge", "round": state["rounds"], "report": run.answer,
                                            **verdict.model_dump(), "messages": run.messages}]}


def build_judge_graph():
    graph = StateGraph(JudgeState)
    graph.add_node("answer", answer_node)
    graph.add_node("judge", judge_node)
    graph.add_edge(START, "answer")
    graph.add_conditional_edges("answer", lambda s: s["route"], {"judge": "judge", "end": END})
    graph.add_conditional_edges("judge", lambda s: s["route"], {"answer": "answer", "end": END})
    return graph.compile()


JUDGE_GRAPH = build_judge_graph()


def solve_judge(task: dict) -> dict[str, Any]:
    state = JUDGE_GRAPH.invoke({"question": task["question"], "answer": "", "first_answer": "", "evidence": "",
                                "feedback": "", "rounds": 0, "steps": 0, "verdicts": [], "history": [],
                                "messages": [], "budget_hit": False, "route": ""},
                               {"recursion_limit": 2 * MAX_ROUNDS + 2})
    first = final_answer(state["first_answer"])
    return {"answer": state["answer"], "messages": state["messages"], "steps": state["steps"],
            "answer_initial": first[:60], "correct_initial": is_correct(task, first), "rounds": state["rounds"],
            "rejections": sum(not v["supported"] for v in state["verdicts"]),
            "first_verdict": state["verdicts"][0]["supported"] if state["verdicts"] else None,
            "budget_hit": state["budget_hit"],
            "trace": {"verdicts": state["verdicts"], "history": state["history"]}}


# --- контроль к судье: повтор при unknown без судьи ---

UNKNOWN_ISSUE = "'unknown' is not an answer."


def after_answer(state: JudgeState) -> str:
    """Тот же потолок раундов, что у судьи; повтор только при unknown — его в судье отклоняет код, а не модель."""
    if state["route"] == "end" or not is_unknown(state["answer"]):
        return "end"
    return "reject"


def reject_unknown_node(state: JudgeState) -> dict[str, Any]:
    """Отказ без вызова модели: замечание то же, что шлёт судья на unknown, но без находок проверяющего."""
    budget_hit = calls_left() < ANSWER_CALLS
    record = {"round": state["rounds"], "answer": final_answer(state["answer"]), "supported": False,
              "issue": UNKNOWN_ISSUE}
    return {"verdicts": state["verdicts"] + [record], "feedback": UNKNOWN_ISSUE, "budget_hit": budget_hit,
            "route": "end" if budget_hit else "answer"}


def build_retry_graph():
    graph = StateGraph(JudgeState)
    graph.add_node("answer", answer_node)
    graph.add_node("reject", reject_unknown_node)
    graph.add_edge(START, "answer")
    graph.add_conditional_edges("answer", after_answer, {"reject": "reject", "end": END})
    graph.add_conditional_edges("reject", lambda s: s["route"], {"answer": "answer", "end": END})
    return graph.compile()


RETRY_GRAPH = build_retry_graph()


def solve_retry(task: dict) -> dict[str, Any]:
    state = RETRY_GRAPH.invoke({"question": task["question"], "answer": "", "first_answer": "", "evidence": "",
                                "feedback": "", "rounds": 0, "steps": 0, "verdicts": [], "history": [],
                                "messages": [], "budget_hit": False, "route": ""},
                               {"recursion_limit": 2 * MAX_ROUNDS + 2})
    first = final_answer(state["first_answer"])
    return {"answer": state["answer"], "messages": state["messages"], "steps": state["steps"],
            "answer_initial": first[:60], "correct_initial": is_correct(task, first), "rounds": state["rounds"],
            "rejections": len(state["verdicts"]), "budget_hit": state["budget_hit"],
            "trace": {"verdicts": state["verdicts"], "history": state["history"]}}


# --- голосование и каскад ---

class VoteState(TypedDict):
    question: str
    votes: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    agreement: float
    escalated: bool
    escalation: dict[str, Any]
    budget_hit: bool


def bag(answer: str) -> str:
    """Короткий ответ как набор слов после нормализации answer_check: «15 March» == «March 15»."""
    return " ".join(sorted(answer_check.answer_tokens(final_answer(answer))))


def fan_out(state: VoteState) -> list[Send]:
    return [Send("vote", {"question": state["question"], "i": i}) for i in range(VOTES)]


def vote_node(arg: dict[str, Any]) -> dict[str, Any]:
    """Один независимый прогон дешёвой модели; голоса идут параллельно."""
    try:
        run = agent(arg["question"], CHEAP, TOOL_NAMES, MAX_STEPS, temperature=VOTE_TEMPERATURE)
        vote = {"i": arg["i"], "answer": run.answer, "steps": run.steps, "messages": run.messages}
    except Exception as e:
        log.exception("голос %s упал", arg["i"])
        vote = {"i": arg["i"], "answer": f"{ERROR_PREFIX} {e}", "steps": 0, "messages": []}
    return {"votes": [vote]}


def tally_node(state: VoteState) -> dict[str, Any]:
    counts = Counter(bag(v["answer"]) for v in state["votes"] if not v["answer"].startswith(ERROR_PREFIX))
    top, count = counts.most_common(1)[0] if counts else ("", 0)
    winner = next((v["answer"] for v in state["votes"] if bag(v["answer"]) == top), "FINAL: unknown")
    consensus = count >= 2 and not is_unknown(top)
    budget_hit = not consensus and calls_left() < ANSWER_CALLS
    return {"answer": winner, "agreement": count / VOTES, "escalated": not consensus and not budget_hit,
            "budget_hit": budget_hit}


def escalate_node(state: VoteState) -> dict[str, Any]:
    run = agent(state["question"], MID, TOOL_NAMES, MAX_STEPS)
    return {"answer": run.answer, "escalation": {"answer": run.answer, "steps": run.steps, "messages": run.messages}}


def build_vote_graph():
    graph = StateGraph(VoteState)
    graph.add_node("vote", vote_node)
    graph.add_node("tally", tally_node)
    graph.add_node("escalate", escalate_node)
    graph.add_conditional_edges(START, fan_out, ["vote"])
    graph.add_edge("vote", "tally")
    graph.add_conditional_edges("tally", lambda s: "escalate" if s["escalated"] else END, ["escalate", END])
    graph.add_edge("escalate", END)
    return graph.compile()


VOTE_GRAPH = build_vote_graph()


def solve_vote(task: dict) -> dict[str, Any]:
    state = VOTE_GRAPH.invoke({"question": task["question"], "votes": [], "answer": "", "agreement": 0.0,
                               "escalated": False, "escalation": {}, "budget_hit": False})
    votes = sorted(state["votes"], key=lambda v: v["i"])
    final = state["escalation"] if state["escalated"] else next(
        (v for v in votes if v["answer"] == state["answer"]), {"messages": []})
    return {"answer": state["answer"], "messages": final["messages"],
            "steps": sum(v["steps"] for v in votes) + state["escalation"].get("steps", 0),
            "votes": " | ".join(final_answer(v["answer"])[:40] for v in votes),
            "agreement": round(state["agreement"], 2), "escalated": state["escalated"],
            "budget_hit": state["budget_hit"],
            "trace": {"vote_runs": votes, "escalation": state["escalation"]}}


PATTERNS = {
    "judge": ("LangGraph + судья, поиск и чтение страницы", solve_judge),
    "vote": ("голосование 3 + каскад haiku, поиск и чтение страницы", solve_vote),
    "retry": ("LangGraph + повтор при unknown, поиск и чтение страницы", solve_retry),
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="паттерны недели поверх агента на LangGraph")
    parser.add_argument("--pattern", required=True, choices=list(PATTERNS))
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--budget", type=int, default=None, help="потолок вызовов модели на вопрос")
    parser.add_argument("--tag", default=None, help="метка конфигурации вместо стандартной")
    args = parser.parse_args()
    assert os.getenv("OPENROUTER_API_KEY"), "нет ключа: положите OPENROUTER_API_KEY в .env"
    if args.budget:
        BUDGET["limit"] = args.budget

    tag, solve = PATTERNS[args.pattern]
    tag = args.tag or tag
    df = run_tasks(load_tasks(limit=args.limit), solve, tag, CHEAP.split("/")[-1])
    df.to_csv(RESULTS / f"{tag}__cheap.csv", index=False, encoding="utf-8")
    print(summary_line(df))
    if args.pattern in ("judge", "retry"):
        fixed = int((~df["correct_initial"] & df["correct"]).sum())
        broken = int((df["correct_initial"] & ~df["correct"]).sum())
        print(f"{args.pattern}: исправил {fixed}, испортил {broken}, отклонений {int(df['rejections'].sum())}, "
              f"упёрлись в потолок {int(df['budget_hit'].sum())}")
    else:
        print(f"наверх ушло {int(df['escalated'].sum())} из {len(df)} ({df['escalated'].mean():.0%}), "
              f"упёрлись в потолок {int(df['budget_hit'].sum())}")
