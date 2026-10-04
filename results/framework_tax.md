# Налог фреймворка на пустом запросе

Вопрос: «Which country will host the 2026 FIFA World Cup final?», модель openai/gpt-4o-mini, первый вызов.

| | голый цикл | LangGraph |
| --- | --: | --: |
| prompt_tokens из usage | 242 | 242 |
| байт в теле запроса | 1464 | 1421 |
| ключи тела | model, messages, usage, tools, tool_choice | model, stream, messages, tool_choice, tools, usage |

Тела совпадают байт в байт: **False**; как JSON: **False**; порядок ключей тот же: **False**; как JSON без `stream`: **True**.

Байты: requests пишет JSON с пробелами после `,` и `:` — без них тело голого цикла 1406 байт; LangGraph компактен и добавляет `"stream":false,` (15 байт).

Разница тел как JSON (ключи отсортированы):

```diff
--- голый цикл
+++ LangGraph
@@ -12,2 +12,3 @@
  "model": "openai/gpt-4o-mini",
+ "stream": false,
  "tool_choice": "auto",
```

Разница заголовков (без Authorization, Content-Length, Host и cookie Cloudflare):

| заголовок | голый цикл | LangGraph |
| --- | --- | --- |
| accept | */* | application/json |
| accept-encoding | gzip, deflate | gzip, deflate, zstd |
| user-agent | python-requests/2.34.2 | OpenAI/Python 3.24.0 |
| x-stainless-arch | — | x64 |
| x-stainless-async | — | false |
| x-stainless-lang | — | python |
| x-stainless-os | — | Linux |
| x-stainless-package-version | — | 3.24.0 |
| x-stainless-raw-response | — | true |
| x-stainless-retry-count | — | 0 |
| x-stainless-runtime | — | CPython |
| x-stainless-runtime-version | — | 3.11.15 |
| x-title | agents-course-seminar01 | agents-course-seminar03 |

Вызовов модели у LangGraph на этом вопросе: 2.
