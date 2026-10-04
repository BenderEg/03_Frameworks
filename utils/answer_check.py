"""Проверка ответа агента против эталона.

Правило семинара — «нормализованный эталон входит в ответ» — сохранено, но нормализация
стала пословной и терпимее к форме:

1. регистр, диакритика (Tomáš Souček == Tomas Soucek), пунктуация и артикли a/an/the
   не учитываются;
2. запятые-разделители тысяч убираются: 64,478 == 64478;
3. числительные и порядковые до двадцати становятся цифрами: two == 2, sixth == 6th == 6;
4. эталон ищется как непрерывная последовательность слов ответа, а не подстрока:
   «6» больше не находится внутри «16»;
5. если эталон — число с единицей измерения (44 years, 84th minute), достаточно
   этого числа отдельным словом в ответе.

Перевод не засчитывается: «Криштиану Роналду» не равно «Cristiano Ronaldo». Поэтому
промпт агента требует FINAL на английском, дословно как в источнике.
"""

import re
import unicodedata

_CARDINALS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
              "fifteen sixteen seventeen eighteen nineteen twenty").split()
_ORDINALS = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
             "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()
NUMBER_WORDS = {word: str(i) for words in (_CARDINALS, _ORDINALS) for i, word in enumerate(words)}
ARTICLES = {"a", "an", "the"}
UNIT_WORDS = {"year", "years", "minute", "minutes", "goal", "goals", "second", "seconds",
              "yard", "yards", "time", "times"}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def answer_tokens(text: str) -> list[str]:
    """Слова ответа после нормализации: без регистра, диакритики, артиклей; числа цифрами."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", _strip_accents(str(text).lower()))
    tokens: list[str] = []
    for word in re.findall(r"\w+", text):
        if word in ARTICLES:
            continue
        word = NUMBER_WORDS.get(word, word)
        tokens.append(re.sub(r"^(\d+)(?:st|nd|rd|th)$", r"\1", word))
    return tokens


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    size = len(needle)
    return any(haystack[i:i + size] == needle for i in range(len(haystack) - size + 1))


def is_correct(gold: str, answer: str) -> bool:
    """Эталон засчитан, если его слова идут подряд в ответе; для «числа с единицей» хватает числа."""
    gold_tokens, answer_tokens_ = answer_tokens(gold), answer_tokens(answer)
    if not gold_tokens:
        return False
    if _contains_run(answer_tokens_, gold_tokens):
        return True
    core = [token for token in gold_tokens if token not in UNIT_WORDS]
    return len(core) == 1 and core != gold_tokens and core[0].isdigit() and core[0] in answer_tokens_
