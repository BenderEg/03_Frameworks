"""Итоговая таблица и картинка «цена вопроса против доли верных» по всем конфигурациям. Без вызовов модели.

Голый цикл одним прогоном через счётчик (bare_loop.py), паттерны и тот же агент на haiku и sonnet — из results/.
Составная строка голого цикла 01_ReAct (повтор только провалов) и sonnet без инструментов — из baseline/
(копия трейсов и CSV 01_ReAct), для справки.
Верность везде пересчитана по полному FINAL из трейсов локальной answer_check (с правилом 6); исходная проверка
из 01_ReAct нужна только для раздела «Проверка проверки».
В 01_ReAct токены не сохранялись, поэтому для этих двух строк они оценены tiktoken по трейсам (помечены ≈);
точность оценки проверяется на трейсах LangGraph, где есть точные токены из счётчика.

    uv run python report.py
"""

import csv, json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import pandas as pd
import tiktoken

from langgraph_agent import (BASELINE, ERROR_PREFIX, FINISH, HERE, RESULTS, SYSTEM, TRACES, final_answer, is_correct,
                             load_tasks)
from patterns import PATTERNS, bag, is_unknown
from utils import answer_check as original_check  # копия проверки 01_ReAct до правки, только для сравнения

N = 50
IMG = HERE / "img"
LANGGRAPH = "LangGraph, поиск и чтение страницы"
JUDGE, VOTE, RETRY = PATTERNS["judge"][0], PATTERNS["vote"][0], PATTERNS["retry"][0]
JUDGE_V1 = "LangGraph + судья v1, поиск и чтение страницы"
BARE = "голый цикл, поиск и чтение страницы"  # голый цикл 01_ReAct одним прогоном через счётчик (bare_loop.py)
HAIKU = "LangGraph на haiku-4.5, поиск и чтение страницы"
SONNET = "LangGraph на sonnet-4.6, поиск и чтение страницы"
REACT_CHEAP = [BASELINE / "traces/повторные запуски/поиск и чтение страницы/gpt-4o-mini",
               BASELINE / "traces/поиск и чтение страницы/gpt-4o-mini"]
REACT_CHEAP_CSV = [BASELINE / "results/повторные запуски/поиск и чтение страницы__cheap.csv",
                   BASELINE / "results/поиск и чтение страницы__cheap.csv"]
REACT_STRONG = BASELINE / "traces/без инструментов/claude-sonnet-4.6"
REACT_STRONG_CSV = BASELINE / "results/без инструментов__strong.csv"

ENC = tiktoken.get_encoding("o200k_base")
CALL_OVERHEAD = 8  # обёртка одного вызова инструмента в истории и в ответе


def count(text: str) -> int:
    return len(ENC.encode(text or ""))


def message_tokens(m: dict) -> int:
    """Как OpenAI считает сообщение: обёртка, текст и у вызовов только имя с аргументами."""
    content = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
    calls = sum(CALL_OVERHEAD + count(c["function"]["name"]) + count(c["function"]["arguments"])
                for c in m.get("tool_calls") or [])
    return 4 + count(content) + calls


@lru_cache
def schema_overhead() -> int:
    """Сколько токенов OpenAI тратит на схемы инструментов: точный вход первого вызова LangGraph минус системный
    промпт и вопрос. Схемы те же, что в 01_ReAct, поэтому константа переносится на голый цикл."""
    calls = pd.read_csv(RESULTS / "calls.csv")
    first = calls[calls["tag"] == LANGGRAPH].groupby("qid", sort=False)["prompt"].first()
    tasks = {t["id"]: t for t in load_tasks(limit=N)}
    gaps = [first[qid] - 3 - message_tokens({"content": SYSTEM}) - message_tokens({"content": tasks[qid]["question"]})
            for qid in first.index if qid in tasks]
    return int(pd.Series(gaps).median())


def estimate_tokens(messages: list[dict], tools: bool) -> int:
    """Вход и выход каждого вызова, собранные заново по трейсу.

    Вызов — это каждое сообщение assistant: на входе все сообщения до него и схемы инструментов (до записки FINISH),
    на выходе его текст и вызовы. Ответ с вызовами, выброшенный перед финальным шагом, восстанавливается как вход
    без записки FINISH, но со схемами.
    """
    schemas = schema_overhead() if tools else 0
    total, prefix, finished = 0, 0, False
    for m in messages:
        if m["role"] == "user" and m.get("content") == FINISH:
            total += prefix + schemas + 3
            finished = True
        if m["role"] == "assistant":
            total += prefix + (0 if finished else schemas) + 3 + message_tokens(m) - 4
        prefix += message_tokens(m)
    return total


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def first_existing(paths: list[Path]) -> Path:
    return next(p for p in paths if p.exists())


def was_correct(task: dict, answer: str) -> bool:
    """Вердикт исходной проверки 01_ReAct — до правила 6."""
    return not answer.startswith(ERROR_PREFIX) and original_check.is_correct(task["answer"], answer)


def first_answer(trace: dict) -> str:
    """Первый ответ из истории судьи: в CSV answer_initial обрезан до 60 символов."""
    return next(h["answer"] for h in trace["history"] if h["node"] == "answer")


def react_rows(config: str, trace_dirs: list[Path], csvs: list[Path], tools: bool) -> pd.DataFrame:
    """Строки 01_ReAct: повторный запуск, если он есть, иначе исходный; верность пересчитана по полному FINAL."""
    tables = [read_csv(p) for p in csvs]
    rows = []
    for task in load_tasks(limit=N):
        trace = json.loads(first_existing([d / f"{task['id']}.json" for d in trace_dirs]).read_text(encoding="utf-8"))
        row = next(t[task["id"]] for t in tables if task["id"] in t)
        answer = final_answer(trace["answer"])
        rows.append({"config": config, "id": task["id"], "answer": answer, "gold": task["answer"],
                     "correct": is_correct(task, answer), "correct_old": was_correct(task, answer),
                     "calls": int(row["steps"]), "tokens": estimate_tokens(trace["messages"], tools),
                     "cost": float(trace["cost"]), "steps": int(row["steps"])})
    return pd.DataFrame(rows)


def local_rows(config: str, level: str = "cheap", model_label: str = "gpt-4o-mini") -> pd.DataFrame:
    """Строки из results/; верность пересчитана по полному ответу из трейса, а не взята из CSV прогона."""
    df = pd.read_csv(RESULTS / f"{config}__{level}.csv")
    tasks = {t["id"]: t for t in load_tasks(limit=N)}
    df = df[df["id"].isin(tasks)].copy()
    traces = {i: json.loads((TRACES / config / model_label / f"{i}.json").read_text(encoding="utf-8"))
              for i in df["id"]}
    df["config"] = config
    df["tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    df["answer"] = [final_answer(traces[i]["answer"]) for i in df["id"]]
    df["correct"] = [is_correct(tasks[i], a) for i, a in zip(df["id"], df["answer"])]
    df["correct_old"] = [was_correct(tasks[i], a) for i, a in zip(df["id"], df["answer"])]
    if "correct_initial" in df:
        df["correct_initial"] = [is_correct(tasks[i], final_answer(first_answer(traces[i]))) for i in df["id"]]
    return df


def summarize(df: pd.DataFrame, label: str, estimated: bool) -> dict:
    n, ok, cost = len(df), int(df["correct"].sum()), df["cost"].sum()
    tokens = f"{df['tokens'].mean():.0f}"
    return {"конфигурация": label, "верно": f"{ok}/{n}", "доля верных": round(ok / n, 2),
            "вызовов на вопрос": round(df["calls"].mean(), 1), "токенов на вопрос": ("≈" if estimated else "") + tokens,
            "центов на вопрос": round(cost / n * 100, 3), "центов за верный": round(cost / ok * 100, 3) if ok else None}


def estimator_error() -> float:
    """Ошибка оценки tiktoken на трейсах LangGraph против точных токенов счётчика, в процентах."""
    df = local_rows(LANGGRAPH)
    estimated = sum(estimate_tokens(json.loads((TRACES / LANGGRAPH / "gpt-4o-mini" / f"{i}.json")
                                               .read_text(encoding="utf-8"))["messages"], True) for i in df["id"])
    return (estimated / df["tokens"].sum() - 1) * 100


# Цвет по роли строки (категориальные слоты 1–3 эталонной палитры dataviz, светлая тема).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROLES = {"база": "#2a78d6", "паттерн": "#eb6834", "ориентир": "#1baf7a", "контроль": "#eda100"}


def money_chart(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    x, y = table["центов на вопрос"], table["доля верных"] * 100
    for role, color in ROLES.items():
        part = table["роль"] == role
        ax.scatter(x[part], y[part], s=90, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=role)
    ax.set_xlim(x.min() / 3.5, x.max() * 3.2)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    fig.canvas.draw()
    placed = []
    for _, r in table.sort_values("доля верных", ascending=False).iterrows():
        text = f"{r['коротко']}\n{r['центов за верный']}¢ за верный ответ"
        for dx, dy in [(9, 8), (9, -14), (-9, 8), (-9, -14), (9, 26), (9, -30)]:
            label = ax.annotate(text, (r["центов на вопрос"], r["доля верных"] * 100), xytext=(dx, dy),
                                textcoords="offset points", fontsize=8.5, color=INK, va="center",
                                ha="left" if dx > 0 else "right",
                                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.6, "shrinkA": 1, "shrinkB": 5,
                                            "relpos": (0 if dx > 0 else 1, 0.5)})
            box = label.get_window_extent()
            if not any(box.overlaps(b) for b in placed) or dy == -30:
                break
            label.remove()
        placed.append(box)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel("цена вопроса, центы (логарифмическая шкала)", color=INK_2)
    ax.set_ylabel("доля верных, %", color=INK_2)
    ax.set_title(f"Цена вопроса против доли верных, {N} вопросов", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    IMG.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)


def judge_lines(name: str, df: pd.DataFrame) -> list[str]:
    fixed = int((~df["correct_initial"] & df["correct"]).sum())
    broken = int((df["correct_initial"] & ~df["correct"]).sum())
    verdict = pd.crosstab(df["correct_initial"].map({True: "первый ответ верен", False: "первый ответ неверен"}),
                          df["first_verdict"].map({True: "судья принял", False: "судья отклонил"}))
    verdict.index.name, verdict.columns.name = None, None
    accepted_unknown = int((df["first_verdict"].eq(True) & df["answer_initial"].map(is_unknown)).sum())
    return [f"### {name}", "",
            f"Исправил **{fixed}**, испортил **{broken}** (первый ответ против итогового в том же прогоне); "
            f"отклонений всего {int(df['rejections'].sum())}, вопросов со второй и третьей попыткой "
            f"{int((df['rounds'] > 1).sum())}, принял unknown с первого раза {accepted_unknown}, "
            f"упёрлись в потолок вызовов {int(df['budget_hit'].sum())}.", "",
            "Первый вердикт судьи против проверки ответа:", "", verdict.to_markdown(), ""]


def capped_example(lg: pd.DataFrame) -> list[str]:
    """Вопрос, где сработал max_steps: что агент ответил после обрыва."""
    capped = lg[lg["steps"] > 8]
    known = capped[~capped["answer"].map(is_unknown)]
    row = (known if len(known) else capped).iloc[0]
    trace = json.loads((TRACES / LANGGRAPH / "gpt-4o-mini" / f"{row['id']}.json").read_text(encoding="utf-8"))
    return [f"У LangGraph потолок max_steps=8 сработал на {len(capped)} вопросах из {len(lg)}, "
            f"из них {int(capped['answer'].map(is_unknown).sum())} закончились ответом unknown. "
            f"Пример {row['id']}: {row['tool_calls']} вызовов инструментов, затем финальный вызов без инструментов.", "",
            f"> {trace['task']['question']}", "",
            f"Ответ после обрыва: `{final_answer(trace['answer'])}`, эталон `{row['gold']}`."]


# «Проверка проверки»: вердикт, вынесенный глазами, для каждого не-unknown ответа, который исходная проверка сочла
# неверным. Ключ — id вопроса, затем (подстрока ответа в нижнем регистре, вердикт); первая совпавшая пара выигрывает.
MISS, DEBATABLE, WRONG = "проверка ошиблась", "спорно, оставлено неверным", "неверно"
REVIEW: dict[str, list[tuple[str, str, str]]] = {
    "fresh-1": [("", MISS, "тот же 1,000-й матч, слова в другом порядке; исправлено правилом 6")],
    "fresh-30": [("1994", MISS, "тот же ЧМ-1994 без слова FIFA; исправлено правилом 6"),
                 ("", WRONG, "ЧМ-2022 вместо ЧМ-1994")],
    "fresh-2": [("", WRONG, "1986 — год, когда тренеры играли друг против друга; прошлый матч открытия — 2010, 16 лет")],
    "fresh-4": [("knockout stage", DEBATABLE, "round of 32 и есть первый раунд плей-офф, но эталон назван точнее"),
                ("", WRONG, "квалификация на сам турнир, а не в round of 32")],
    "fresh-6": [("round of 32", DEBATABLE, "round of 32 — первый раунд плей-офф, но эталон «knockout stages»"),
                ("", WRONG, "round of 16: в 2026 первый раунд плей-офф — 1/16")],
    "fresh-12": [("abunada", DEBATABLE, "только фамилия, а промпт требует имя как в источнике"),
                 ("", WRONG, "автогол забил другой игрок")],
    "fresh-21": [("burgess", DEBATABLE, "только фамилия, а промпт требует имя как в источнике"),
                 ("", WRONG, "Бобадилья — автогол в матче США — Парагвай, а в матче с Австралией — Кэмерон Бёрджесс")],
    "fresh-44": [("", DEBATABLE, "только фамилия, а промпт требует имя как в источнике")],
    "fresh-14": [("", WRONG, "прошлая победа Шотландии — 1990 (Швеция 2–1), эталон совпадает с источником")],
    "fresh-28": [("", WRONG, "вопрос о первом очке Кюрасао, а не о первом участии")],
    "fresh-39": [("", WRONG, "прошлое поражение — в 1982, 44 года назад; эталон совпадает с источником")],
    "fresh-48": [("", WRONG, "67 — сегодняшний счёт Мбаппе из статьи о нём, а на момент рекорда было 58")],
}


def review(qid: str, answer: str) -> tuple[str, str]:
    return next((verdict, note) for part, verdict, note in REVIEW[qid] if part in answer.lower())


def check_lines(frames: dict[str, pd.DataFrame], mini: list[str]) -> list[str]:
    """Сколько ответов исходная проверка зря сочла неверными: счёт до и после правки и прочитанные ответы."""
    before_after = pd.DataFrame([{
        "конфигурация": name, "неверных до": int((~df["correct_old"]).sum()), "из них unknown":
        int((~df["correct_old"] & df["answer"].map(is_unknown)).sum()),
        "верно до": int(df["correct_old"].sum()), "верно после": int(df["correct"].sum())} for name, df in frames.items()])
    read = []
    for name in mini:
        df = frames[name]
        for _, r in df[~df["correct_old"] & ~df["answer"].map(is_unknown)].iterrows():
            verdict, note = review(r["id"], r["answer"])
            read.append({"конфигурация": name, "id": r["id"], "ответ": r["answer"][:70], "эталон": r["gold"],
                         "вердикт": verdict, "почему": note, "после правки": "+" if r["correct"] else "−"})
    read = pd.DataFrame(read)
    verdicts = read.groupby("конфигурация", sort=False)["вердикт"].value_counts().unstack(fill_value=0)
    verdicts = verdicts.reindex(columns=[MISS, DEBATABLE, WRONG], fill_value=0)
    verdicts.columns.name = None
    return ["## Проверка проверки", "", before_after.to_markdown(index=False), "",
            "Прочитаны глазами все не-unknown ответы gpt-4o-mini, которые исходная проверка сочла неверными:", "",
            verdicts.to_markdown(), "", read.to_markdown(index=False), ""]


def votes_only(tag: str) -> tuple[int, int]:
    """Сколько верных дал бы один голос и большинство из трёх без каскада — по тем же трейсам, без вызовов."""
    single = plural = 0
    for task in load_tasks(limit=N):
        trace = json.loads((TRACES / tag / "gpt-4o-mini" / f"{task['id']}.json").read_text(encoding="utf-8"))
        votes = [v["answer"] for v in trace["vote_runs"]]
        top = Counter(bag(a) for a in votes).most_common(1)[0][0]
        single += is_correct(task, final_answer(votes[0]))
        plural += is_correct(task, final_answer(next(a for a in votes if bag(a) == top)))
    return single, plural


def fixed_ids(df: pd.DataFrame) -> set[str]:
    return set(df.loc[~df["correct_initial"] & df["correct"], "id"])


def control_lines(judge: pd.DataFrame, retry: pd.DataFrame) -> list[str]:
    """Судья против повтора при unknown без судьи: окупает ли проверяющий агент свою цену."""
    rows = []
    for name, df in (("повтор при unknown", retry), ("судья", judge)):
        ok, cost = int(df["correct"].sum()), df["cost"].sum()
        rows.append({"конфигурация": name, "верно": f"{ok}/{len(df)}", "исправил": len(fixed_ids(df)),
                     "испортил": int((df["correct_initial"] & ~df["correct"]).sum()),
                     "вопросов с повтором": int((df["rounds"] > 1).sum()),
                     "центов на вопрос": round(cost / len(df) * 100, 3),
                     "центов за верный": round(cost / ok * 100, 3) if ok else None})
    both = fixed_ids(judge) & fixed_ids(retry)
    only_judge, only_retry = fixed_ids(judge) - both, fixed_ids(retry) - both
    unknown_first = int(retry["answer_initial"].fillna("").map(is_unknown).sum())
    return ["## Судья против повтора при unknown", "",
            pd.DataFrame(rows).to_markdown(index=False), "",
            f"Первый ответ unknown у повтора: {unknown_first} из {len(retry)}. "
            f"Из {len(fixed_ids(judge))} вопросов, исправленных судьёй, повтор без судьи исправил тоже {len(both)}"
            f"{': ' + ', '.join(sorted(both, key=qnum)) if both else ''}. "
            f"Только судья: {', '.join(sorted(only_judge, key=qnum)) or 'нет'}; "
            f"только повтор: {', '.join(sorted(only_retry, key=qnum)) or 'нет'}.", ""]


def qnum(qid: str) -> int:
    return int(qid.split("-")[-1])


def strong_lines(table: pd.DataFrame, vote: pd.DataFrame, haiku: pd.DataFrame) -> list[str]:
    """Каскад против того же агента на haiku и sonnet по всем вопросам: сколько экономят голоса."""
    rows = table.set_index("коротко")
    cascade, h, s = rows.loc["голосование + каскад"], rows.loc["LangGraph haiku"], rows.loc["LangGraph sonnet"]
    escalated = vote.loc[vote["escalated"], "id"]
    h_on_escalated = haiku[haiku["id"].isin(escalated)]
    return ["## Каскад против того же агента на сильной модели", "",
            rows.loc[["голосование + каскад", "LangGraph haiku", "LangGraph sonnet"],
                     ["верно", "центов на вопрос", "центов за верный"]].to_markdown(), "",
            f"Каскад стоит {cascade['центов на вопрос'] / h['центов на вопрос']:.0%} цены haiku на всех вопросах "
            f"и {cascade['центов на вопрос'] / s['центов на вопрос']:.0%} цены sonnet. "
            f"На {len(escalated)} вопросах, ушедших наверх, haiku в своём прогоне верна "
            f"{int(h_on_escalated['correct'].sum())}/{len(h_on_escalated)} "
            f"(в каскаде {int(vote.loc[vote['escalated'], 'correct'].sum())}/{len(escalated)}), "
            f"на остальных {int(haiku.loc[~haiku['id'].isin(escalated), 'correct'].sum())}/{len(haiku) - len(escalated)} "
            f"против {int(vote.loc[~vote['escalated'], 'correct'].sum())}/{len(vote) - len(escalated)} у большинства голосов mini.", ""]


# Какие конфигурации на gpt-4o-mini прочитаны глазами в «Проверке проверки».
READ = ["голый цикл", "LangGraph", "повтор при unknown", "судья v1", "LangGraph + судья", "голосование + каскад",
        "голый цикл 01_ReAct"]


def main() -> None:
    configs = [
        ("голый цикл, gpt-4o-mini (один прогон через счётчик)", "голый цикл", "база", False, local_rows(BARE)),
        ("LangGraph, gpt-4o-mini", "LangGraph", "база", False, local_rows(LANGGRAPH)),
        ("LangGraph + повтор при unknown (контроль к судье)", "повтор при unknown", "контроль", False,
         local_rows(RETRY)),
        ("LangGraph + судья v1 (принимал unknown)", "судья v1", "паттерн", False, local_rows(JUDGE_V1)),
        ("LangGraph + судья, gpt-4o-mini", "LangGraph + судья", "паттерн", False, local_rows(JUDGE)),
        ("голосование 3 + каскад на haiku-4.5", "голосование + каскад", "паттерн", False, local_rows(VOTE)),
        ("LangGraph на claude-haiku-4.5", "LangGraph haiku", "ориентир", False,
         local_rows(HAIKU, "mid", "claude-haiku-4.5")),
        ("LangGraph на claude-sonnet-4.6", "LangGraph sonnet", "ориентир", False,
         local_rows(SONNET, "strong", "claude-sonnet-4.6")),
        ("голый цикл 01_ReAct: повтор только провалов", "голый цикл 01_ReAct", "ориентир", True,
         react_rows("голый цикл 01_ReAct", REACT_CHEAP, REACT_CHEAP_CSV, tools=True)),
        ("claude-sonnet-4.6 без инструментов (01_ReAct)", "sonnet без инструментов", "ориентир", True,
         react_rows("sonnet", [REACT_STRONG], [REACT_STRONG_CSV], tools=False)),
    ]
    table = pd.DataFrame([{**summarize(df, label, est), "коротко": short, "роль": role}
                          for label, short, role, est, df in configs])
    money_chart(table, IMG / "money_quality.png")

    frames = {short: df for _, short, _, _, df in configs}
    vote = frames["голосование + каскад"]
    calls = pd.read_csv(RESULTS / "calls.csv")
    calls = calls[calls["tag"] == VOTE]
    haiku_share = calls.loc[calls["model"].str.contains("haiku"), "cost"].sum() / calls["cost"].sum()
    mini_cents = calls.loc[~calls["model"].str.contains("haiku"), "cost"].sum() / N * 100
    single, plural = votes_only(VOTE)
    agreed, escalated = vote[~vote["escalated"]], vote[vote["escalated"]]

    lines = [
        f"# Итог на {N} вопросах", "", table.drop(columns=["коротко", "роль"]).to_markdown(index=False), "",
        "≈ — токены оценены tiktoken по трейсам 01_ReAct (счётчик там токены не сохранял); "
        f"на трейсах LangGraph оценка расходится с точным счётчиком на {estimator_error():+.1f}%.", "",
        "## Судья", "",
        *judge_lines("Судья (unknown всегда отклоняется кодом)", frames["LangGraph + судья"]),
        *judge_lines("Судья v1 (unknown решала модель)", frames["судья v1"]),
        *control_lines(frames["LangGraph + судья"], frames["повтор при unknown"]),
        "## Голосование и каскад", "",
        f"Наверх ушло **{len(escalated)} из {len(vote)} ({len(escalated) / len(vote):.0%})**. "
        f"Где большинство нашлось, верно {int(agreed['correct'].sum())}/{len(agreed)}; "
        f"из ушедших на haiku верно {int(escalated['correct'].sum())}/{len(escalated)}. "
        f"На haiku пришлось {haiku_share:.0%} цены конфигурации; упёрлись в потолок вызовов "
        f"{int(vote['budget_hit'].sum())}.", "",
        f"Без каскада по тем же трейсам: один голос mini верен {single}/{N}, большинство из трёх {plural}/{N} "
        f"за {mini_cents:.3f}¢ на вопрос — прирост даёт haiku на спорных вопросах, а голоса работают как маршрутизатор.", "",
        *strong_lines(table, vote, frames["LangGraph haiku"]),
        "## Потолок на деньги", "", *capped_example(frames["LangGraph"]), "",
        *check_lines(frames, READ),
    ]
    report = "\n".join(lines)
    (RESULTS / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
