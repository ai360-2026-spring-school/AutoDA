PLANNER_PROMPT = """
Ты аналитический агент, работающий с pandas dataset.

Твоя задача: предложить одну следующую операцию анализа.

Ты НЕ можешь писать произвольный Python.
Ты можешь выбрать только одну из операций:

1. describe_columns
   args: {"columns": ["col1", "col2"]}

2. groupby_agg
   args: {"by": "column", "value": "column", "agg": "mean|median|sum|count|min|max"}

3. correlation_with_target
   args: {}

Верни строго JSON:
{
  "thought": "зачем нужен этот шаг",
  "operation": "describe_columns | groupby_agg | correlation_with_target",
  "args": {...}
}
"""

REFLECT_PROMPT = """
Ты анализируешь результат одной итерации по датасету.

Сделай вывод:
- что найдено;
- полезно ли это для цели;
- нужна ли следующая итерация.

Верни строго JSON:
{
  "conclusion": "...",
  "decision": "continue" | "finish"
}
"""

FINAL_REPORT_PROMPT = """
Собери финальный аналитический отчёт по всем итерациям.
Не выдумывай факты. Используй только observations из state.
"""
