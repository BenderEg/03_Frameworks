"""Тесты локальной answer_check: случаи из 01_ReAct/tests/test_answer_check.py плюс правило 6 из «Проверки проверки».

Спорные ответы (только фамилия, round of 32 против knockout stages) по-прежнему не засчитываются.
"""

import pytest

from answer_check import answer_tokens, is_correct
from langgraph_agent import is_correct as task_is_correct


@pytest.mark.parametrize(("gold", "answer"), [
    ("two", "2"),
    ("sixth", "6"),
    ("sixth", "6th"),
    ("fourteen", "14 matches"),
    ("44 years", "44"),
    ("84th minute", "84"),
    ("64,478", "64478 spectators"),
    ("Tomas Soucek", "Tomáš Souček"),
    ("United States", "the United States."),
    ("90+7'", "90+7"),
    ("1,000th World Cup match", "The 1,000th World Cup match"),
    # правило 6: число в окружении FIFA/World Cup/match/edition
    ("1,000th World Cup match", "The Group F game between Tunisia and Japan was the 1,000th match in FIFA World Cup history."),
    ("1994 FIFA World Cup", "1994 World Cup"),
    ("1994 FIFA World Cup", "the 1994 edition."),
])
def test_accepts_same_answer_in_another_form(gold: str, answer: str) -> None:
    assert is_correct(gold, answer)


@pytest.mark.parametrize(("gold", "answer"), [
    ("6", "16"),
    ("two", "12"),
    ("1958", "1954"),
    ("44 years", "4 years"),
    ("44 years", "144"),
    ("Cristiano Ronaldo", "Криштиану Роналду"),
    ("Toronto", "unknown"),
    ("", "anything"),
    # правило 6 не распространяется на другие числа и слова
    ("1994 FIFA World Cup", "2026 FIFA World Cup"),
    ("1,000th World Cup match", "900th World Cup match"),
    ("1990", "1998"),
    # спорные: оставлены неверными (промпт требует имя как в источнике; эталон говорит knockout stages)
    ("Cameron Burgess", "Burgess"),
    ("Hassan Al-Tambakti", "Al-Tambakti"),
    ("knockout stages", "Round of 32."),
    ("round of 32", "the knockout stage"),
])
def test_rejects_wrong_answer(gold: str, answer: str) -> None:
    assert not is_correct(gold, answer)


def test_answer_tokens_normalize_form() -> None:
    assert answer_tokens("The Sixth goal, 64,478 fans — Souček") == ["6", "goal", "64478", "fans", "soucek"]


def test_failed_run_is_never_correct() -> None:
    assert not task_is_correct({"answer": "10"}, 'ошибка: HTTP 429 : {"error": {"code": 10}}')
