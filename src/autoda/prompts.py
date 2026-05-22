from __future__ import annotations

import json
from typing import Any

from .actions.registry import catalog_text as _catalog_text

# Generated once at import time — never changes during a run
_CATALOG_COMPACT: str = _catalog_text()

# ---------------------------------------------------------------------------
# Description summarisation (runs once)
# ---------------------------------------------------------------------------

DESCRIPTION_SUMMARISE_PROMPT = """\
Ты — ассистент по машинному обучению. Пользователь предоставил описание датасета и задачи.

ОПИСАНИЕ ПОЛЬЗОВАТЕЛЯ:
{description}

Твоя задача — создать ДВА разных резюме в формате JSON:

1. "long_summary" — детальное резюме для эксперта-критика (до 1000 символов):
   - Домен и смысл задачи
   - Что означают ключевые колонки
   - Важные ограничения или нюансы данных
   - Что предсказываем и почему это важно

2. "short_summary" — сверхкороткое резюме для планировщика (до 300 символов):
   - 2-4 предложения: задача, таргет, самые важные признаки

Ответь СТРОГО в формате JSON без markdown и пояснений:
{{"long_summary": "...", "short_summary": "..."}}"""


def build_description_summarise_prompt(description: str) -> str:
    return DESCRIPTION_SUMMARISE_PROMPT.format(description=description[:3000])


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_PLANNER_BASE = """\
Ты — агент по разработке признаков (feature engineering) для ML-соревнований.

ЗАДАЧА: {task} | МЕТРИКА: {metric_name} ({metric_direction}) | ТАРГЕТ: {target}
ЦЕЛЬ: {goal}

=== ОПИСАНИЕ ЗАДАЧИ (короткое) ===
{short_summary}

╔══════════════════════════════════════════════════════════╗
║  ДИРЕКТИВА КРИТИКА — ОБЯЗАТЕЛЬНО ВЫПОЛНИ НА ЭТОМ ШАГЕ  ║
╚══════════════════════════════════════════════════════════╝
{critic_message}

┌─────────────────────────────────────────────────────────┐
│  ПРЕДЛОЖЕНИЯ ИДЕАТОРОВ — ВЫПОЛНИ HIGH-PRIORITY ПЕРВЫМ  │
│  Если есть high → реализуй его, не придумывай своё.    │
│  Если high нет или всё уже пробовал — действуй сам.    │
└─────────────────────────────────────────────────────────┘
{ideator_suggestions}

=== ТИПЫ КОЛОНОК ===
{column_type_map}

=== ТВОЯ ПАМЯТЬ ===
{planner_memory}

=== ЛОГ ЭКСПЕРИМЕНТОВ (последние {log_count}) ===
{experiment_log}

=== ТЕКУЩАЯ ИТЕРАЦИЯ (шаг {step}) — уже применено ===
{current_iter_transforms}

=== КАТАЛОГ ОПЕРАЦИЙ ===
{catalog}

=== ЧТО ДАЁТ ПРИРОСТ С CATBOOST, А ЧТО НЕТ ===
CatBoost — дерево решений. Он УЖЕ умеет:
  ✓ Категориальные фичи нативно — OHE вручную НЕ нужен
  ✓ Нелинейность (полиномы, квадраты) — сам строит через сплиты
  ✓ Масштаб и порядок значений — стандартизация НЕ нужна

CatBoost НЕ умеет без явных признаков:
  ✗ Произведение двух непрерывных (A × B) — добавляй явно через interaction
  ✗ Отношение (A / B) — добавляй явно через interaction(op="div")
  ✗ Логарифм сильно скошенной фичи — добавляй явно через log_transform
  ✗ Агрегаты по группе (mean_target_by_category) — добавляй через group_aggregate

=== ДОСТУПНЫЕ ДЕЙСТВИЯ ===
Выбери ОДНО действие и верни строгий JSON.
КРИТИЧНО: используй ТОЧНЫЕ имена аргументов из каталога выше — иначе ошибка.

1. read_info — получить информацию (без изменений датасета):
{{"type": "read_info", "op": "<op_name>", "args": {{...}}}}

2. transform — применить трансформацию:
{{"type": "transform", "op": "<op_name>", "args": {{...}}}}

3. update_memory — записать наблюдение в память:
{{"type": "update_memory", "notes": ["наблюдение"]}}

4. submit — отправить на оценку (CatBoost CV):
{{"type": "submit", "rationale": "обоснование"}}

5. cancel — отменить изменения итерации:
{{"type": "cancel"}}

6. stop — завершить:
{{"type": "stop", "reason": "причина"}}

ВАЖНО:
- Директива критика выше — это КОМАНДА, не совет. Если она есть — выполни её первой.
- Не повторяй уже отклонённые операции из лога
- group_aggregate принимает ОДИН by (str) и ОДИН value (str), не список
- multi_col_lambda: для колонок с пробелами/спецсимволами ОБЯЗАТЕЛЬНО используй col('имя').
  Пример: {{"expression": "col('LapTime (s)') * TyreLife", "input_columns": ["LapTime (s)", "TyreLife"], "result_column": "lap_x_tyre"}}
- ЛИМИТ ТРАНСФОРМАЦИЙ: делай НЕ БОЛЕЕ 2 трансформаций за итерацию, затем submit.
  Пакет из 5+ действий нельзя откатать по частям — если он провалится, непонятно что именно помешало.
  Лучше принять 1 фичу и потерять 0, чем отклонить пакет из 5 хороших фич вместе с 1 плохой.
"""

_PLANNER_ADDENDUM_TEMPLATES: dict[str, str] = {
    "info_result": """\
=== РЕЗУЛЬТАТ INFO-ТУЛА ({op}) ===
{result}
""",
    "transform_applied": """\
=== ТРАНСФОРМАЦИЯ ПРИМЕНЕНА ===
Операция: {op}({args})
Изменённые колонки: {changed_columns}
Обновлённые типы: {new_types}
Все изменения этой итерации: {all_transforms}
""",
    "transform_error": """\
=== ОШИБКА ТРАНСФОРМАЦИИ (все изменения откатаны) ===
Операция: {op}({args})
Ошибка: {error}
""",
    "memory_updated": """\
=== ПАМЯТЬ ОБНОВЛЕНА ===
Все изменения этой итерации: {all_transforms}
""",
    "cancelled": """\
=== ИТЕРАЦИЯ ОТМЕНЕНА ===
Откатано трансформаций: {n}
""",
    "new_iteration": """\
=== НОВАЯ ИТЕРАЦИЯ (шаг {step}) ===
Базовый CV (текущий): {cv:.6f}
Решение по прошлой итерации: {decision} (delta={delta:+.6f})
""",
    "parse_error": """\
=== ОШИБКА ПАРСИНГА ОТВЕТА ===
Ответ должен быть строгим JSON. Попробуй снова.
Ошибка: {error}
""",
}


def build_planner_prompt(state: dict[str, Any]) -> str:
    col_map = state.get("column_type_map", {})
    col_map_str = "\n".join(
        f"  {col}: {kind}" for col, kind in sorted(col_map.items())
    ) or "  (нет данных)"

    memory = state.get("planner_memory", [])
    mem_text = _format_memory(memory)

    exp_log = state.get("experiment_log", [])
    log_text, log_count = _format_experiment_log(exp_log, last_n=20)

    critic = state.get("critic_message") or "(нет)"

    suggestions = state.get("ideator_suggestions", [])
    suggestions_str = format_ideator_suggestions(suggestions)

    iter_transforms = state.get("current_iteration_transforms", [])
    step = state.get("current_step", 0) + 1
    if iter_transforms:
        current_iter_str = _format_iter_transforms(iter_transforms)
    else:
        current_iter_str = "(ничего не применено)"

    base = _PLANNER_BASE.format(
        task=state.get("task", "?"),
        metric_name=state.get("metric_name", "?"),
        metric_direction=state.get("metric_direction", "?"),
        target=state.get("target", "?"),
        goal=state.get("goal", ""),
        short_summary=state.get("short_description_summary") or "(нет описания)",
        column_type_map=col_map_str,
        planner_memory=mem_text,
        experiment_log=log_text,
        log_count=log_count,
        critic_message=critic,
        ideator_suggestions=suggestions_str,
        catalog=_CATALOG_COMPACT,
        step=step,
        current_iter_transforms=current_iter_str,
    )

    addendum = _build_addendum(state.get("planner_addendum"))
    return (base + addendum).strip()


def _format_memory(memory: list[str]) -> str:
    if not memory:
        return "  (пусто)"
    text = "\n".join(f"  - {note}" for note in memory)
    if len(text) > 2000:
        lines = []
        total = 0
        for note in reversed(memory):
            line = f"  - {note}"
            if total + len(line) > 1900:
                break
            lines.append(line)
            total += len(line) + 1
        text = "\n".join(reversed(lines)) + "\n  ... (старые записи обрезаны)"
    return text


def _format_experiment_log(log: list[dict[str, Any]], last_n: int = 20) -> tuple[str, int]:
    recent = log[-last_n:]
    if not recent:
        return "  (нет экспериментов)", 0
    lines = []
    for entry in recent:
        step = entry.get("step", "?")
        decision = entry.get("decision", "?")
        delta = entry.get("delta", 0.0)
        cv_after = entry.get("cv_after", entry.get("cv_before", 0.0))
        rationale = entry.get("rationale", "")
        transforms = entry.get("transforms", [])
        ops_str = "; ".join(
            f'{t.get("op","?")}({json.dumps(t.get("args", {}), ensure_ascii=False)[:60]})'
            for t in transforms
        )
        lines.append(
            f"  [шаг {step}] {decision} delta={delta:+.6f} cv={cv_after:.6f}"
            f" | {ops_str} | {rationale[:80]}"
        )
    return "\n".join(lines), len(recent)


def _build_addendum(addendum: dict[str, Any] | None) -> str:
    if not addendum:
        return ""
    lines = ["\n\n--- РЕЗУЛЬТАТ ПОСЛЕДНЕГО ДЕЙСТВИЯ ---"]

    if addendum.get("info_result") is not None:
        op = addendum.get("info_op", "?")
        # Keep result compact — large results bloat the prompt and cause empty LLM responses.
        result_str = json.dumps(addendum["info_result"], ensure_ascii=False)[:400]
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["info_result"].format(op=op, result=result_str))

    elif addendum.get("transform_applied"):
        info = addendum["transform_applied"]
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["transform_applied"].format(
            op=info.get("op", "?"),
            args=json.dumps(info.get("args", {}), ensure_ascii=False)[:200],
            changed_columns=info.get("changed_columns", []),
            new_types=info.get("new_types", {}),
            all_transforms=_format_iter_transforms(addendum.get("all_transforms", [])),
        ))

    elif addendum.get("transform_error"):
        info = addendum["transform_error"]
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["transform_error"].format(
            op=info.get("op", "?"),
            args=json.dumps(info.get("args", {}), ensure_ascii=False)[:200],
            error=info.get("error", ""),
        ))

    elif addendum.get("memory_updated"):
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["memory_updated"].format(
            all_transforms=_format_iter_transforms(addendum.get("all_transforms", [])),
        ))

    elif addendum.get("cancelled"):
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["cancelled"].format(
            n=addendum.get("rolled_back_n", 0),
        ))

    elif addendum.get("new_iteration"):
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["new_iteration"].format(
            step=addendum.get("step", "?"),
            cv=addendum.get("cv", 0.0),
            decision=addendum.get("decision", "?"),
            delta=addendum.get("delta", 0.0),
        ))

    elif addendum.get("parse_error"):
        lines.append(_PLANNER_ADDENDUM_TEMPLATES["parse_error"].format(
            error=addendum.get("parse_error", ""),
        ))

    return "\n".join(lines)


def _format_iter_transforms(transforms: list[dict[str, Any]]) -> str:
    if not transforms:
        return "(нет)"
    return "; ".join(
        f'{t.get("op","?")}({json.dumps(t.get("args",{}), ensure_ascii=False)[:60]})'
        for t in transforms
    )


# ---------------------------------------------------------------------------
# Ideator roles and prompt
# ---------------------------------------------------------------------------

IDEATOR_ROLES: list[dict[str, str]] = [
    {
        "name": "Математик",
        "instruction": (
            "Ты — Математик. Предложи ОДНУ трансформацию основанную на математических свойствах: "
            "отношения признаков (A/B), произведения непрерывных (A×B), логарифм сильно скошенных, "
            "разности (A−B), корни. Думай о нелинейностях которые дерево не строит само."
        ),
    },
    {
        "name": "Доменный эксперт",
        "instruction": (
            "Ты — Доменный эксперт. Используй описание датасета и здравый смысл предметной области. "
            "Предложи ОДНУ физически или содержательно осмысленную комбинацию признаков. "
            "Например: сила = масса × расстояние, стоимость на кг = цена / вес."
        ),
    },
    {
        "name": "Стратег",
        "instruction": (
            "Ты — Стратег. Внимательно прочитай историю экспериментов. "
            "Найди то, что ЕЩЁ НЕ ПРОБОВАЛИ из доступных признаков. "
            "Предложи ОДНУ идею из принципиально нового направления — не повторяй отклонённые."
        ),
    },
    {
        "name": "Аналитик взаимодействий",
        "instruction": (
            "Ты — Аналитик взаимодействий. Специализируешься на парных взаимодействиях. "
            "Предложи ОДНУ пару признаков, которые вместе несут больше информации чем по отдельности. "
            "Обоснуй почему именно эта пара важна для таргета."
        ),
    },
    {
        "name": "Энкодер",
        "instruction": (
            "Ты — Энкодер. Специализируешься на работе с категориальными и дискретными переменными: "
            "target encoding (group_aggregate), frequency encoding, взаимодействия кат×числовой. "
            "Предложи ОДНУ идею по лучшему использованию категориальных признаков."
        ),
    },
]

_IDEATOR_PROMPT_TEMPLATE = """\
Задача ML: {task}, таргет={target}, метрика={metric_name} ({metric_direction}).
Текущий CV: {current_cv:.4f} (baseline: {baseline_cv:.4f}).
{description_line}

Колонки датасета:
{columns}

История последних экспериментов (REJECTED = не сработало, не повторяй!):
{history}

УЖЕ ОТКЛОНЁННЫЕ операции (ЗАПРЕЩЕНО предлагать снова — priority должен быть low):
{rejected_ops}

{instruction}

ВАЖНО:
- Если предлагаешь операцию из списка "уже отклонённых" — поставь priority: "low"
- Предлагай конкретную пару/тройку колонок с реальными именами из списка выше
- Не предлагай нормализацию/стандартизацию — деревья не нуждаются

Верни СТРОГО JSON без пояснений:
{{"suggestion": "конкретное действие: op(col1, col2)", "rationale": "почему должно помочь", "priority": "high|medium|low"}}"""


def build_ideator_prompt(state: dict[str, Any], role_instruction: str) -> str:
    """Short single-turn prompt for one ideator role."""
    target = state["target"]
    task = state.get("task", "regression")
    metric_name = state.get("metric_name", "cv")
    metric_direction = state.get("metric_direction", "min")
    baseline_cv = state.get("baseline_cv_mean") or 0.0

    # Current CV from last experiment
    exp_log = state.get("experiment_log", [])
    if exp_log:
        current_cv = exp_log[-1].get("cv_after") or baseline_cv
    else:
        current_cv = baseline_cv

    col_map = state.get("column_type_map", {})
    cols_str = "\n".join(
        f"  {col} ({kind})" for col, kind in sorted(col_map.items()) if col != target
    ) or "  (нет данных)"

    short_summary = state.get("short_description_summary", "")
    description_line = f"Описание: {short_summary}" if short_summary else ""

    recent = exp_log[-10:]
    history_lines = []
    for e in recent:
        ops = ", ".join(t.get("op", "?") for t in e.get("transforms", []))
        decision = e.get("decision", "?")
        delta = e.get("delta", 0.0)
        history_lines.append(f"  шаг {e.get('step')}: {decision} ({delta:+.4f}) — {ops}")
    history_str = "\n".join(history_lines) if history_lines else "  (история пуста)"

    # Build compact list of rejected op signatures to explicitly block
    rejected_sigs: set[str] = set()
    for e in exp_log:
        if e.get("decision") == "reject":
            for t in e.get("transforms", []):
                op = t.get("op", "")
                args = t.get("args", {})
                cols = args.get("cols", args.get("columns", []))
                if isinstance(cols, list):
                    sig = f"{op}({','.join(sorted(str(c) for c in cols))})"
                else:
                    sig = f"{op}({cols})"
                rejected_sigs.add(sig)
    rejected_ops_str = "\n".join(f"  - {s}" for s in sorted(rejected_sigs)) or "  (нет)"

    return _IDEATOR_PROMPT_TEMPLATE.format(
        task=task,
        target=target,
        metric_name=metric_name,
        metric_direction=metric_direction,
        current_cv=current_cv,
        baseline_cv=baseline_cv,
        description_line=description_line,
        columns=cols_str,
        history=history_str,
        rejected_ops=rejected_ops_str,
        instruction=role_instruction,
    )


def format_ideator_suggestions(suggestions: list[dict[str, Any]]) -> str:
    """Format ideator suggestions for inclusion in planner prompt."""
    if not suggestions:
        return "(нет предложений от идеаторов)"
    lines = []
    for s in suggestions:
        role = s.get("role", "?")
        suggestion = s.get("suggestion", "")
        rationale = s.get("rationale", "")
        priority = s.get("priority", "medium")
        lines.append(f"  [{role}] ({priority}) {suggestion} — {rationale}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Critic prompt
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """\
Ты — супервизор агента по разработке признаков. Модель оценки — CatBoost (деревья решений).

ОПИСАНИЕ ЗАДАЧИ:
{long_summary}

ТАРГЕТ (его нельзя трансформировать): {target}

ДОСТУПНЫЕ КОЛОНКИ В ДАТАСЕТЕ (используй ТОЛЬКО эти имена):
{column_type_map}

ЛОГ ВСЕХ ЭКСПЕРИМЕНТОВ (последние {log_count}):
{experiment_log}

Твоя задача — дать агенту КОНКРЕТНУЮ ДИРЕКТИВУ на следующий шаг.

Правила CatBoost, которые нарушает агент:
- Нормализация/стандартизация — бесполезна для деревьев, не предлагай
- Полиномы (age², age³) — деревья строят нелинейность сами, польза минимальна
- OHE вручную — CatBoost обрабатывает категориальные нативно
- НЕЛЬЗЯ трансформировать таргет ({target})

Что действительно помогает CatBoost:
- Отношения двух непрерывных (A/B): cement/water, bmi/age, distance/weight
- Произведения (A×B): особенно когда обе фичи значимы по отдельности
- Логарифм сильно скошенных фич (skew > 2)
- Агрегаты по категориальной группе (mean_target_by_category)

Анализируй лог и выбери ОДИН из трёх вариантов ответа:

1. Если агент застрял (3+ отклонений одного типа) — напиши прямую команду:
   "Прекрати [X]. Следующий шаг: [конкретная операция, используя ТОЛЬКО существующие колонки из списка выше]."

2. Если агент игнорирует важную область из описания задачи — напиши:
   "Ещё не исследовано: [конкретная фича]. Попробуй [операция с реальными именами колонок]."

3. Если поведение разумное — верни null.

Ответь СТРОГО в формате JSON: {{"message": "директива" или null}}"""


def build_critic_prompt(state: dict[str, Any]) -> str:
    long_summary = (state.get("long_description_summary") or "(описание не задано)")[:1000]
    exp_log = state.get("experiment_log", [])
    log_text, log_count = _format_experiment_log(exp_log, last_n=20)
    target = state.get("target", "target")
    col_map = state.get("column_type_map", {})
    col_map_str = ", ".join(
        f"{col}({kind})" for col, kind in sorted(col_map.items())
    ) or "(нет данных)"
    return CRITIC_PROMPT.format(
        long_summary=long_summary,
        target=target,
        column_type_map=col_map_str,
        experiment_log=log_text,
        log_count=log_count,
    )


# ---------------------------------------------------------------------------
# Final report prompt
# ---------------------------------------------------------------------------

FINAL_REPORT_PROMPT = """\
Составь краткий отчёт о проделанной работе агента по разработке признаков.

ЗАДАЧА: {task} | МЕТРИКА: {metric_name} | ТАРГЕТ: {target}
ЦЕЛЬ: {goal}

Базовый CV: {baseline_cv:.6f}
Итоговый CV: {final_cv:.6f}
Всего итераций: {n_iterations}
Применено изменений: {n_kept}
Отклонено изменений: {n_rejected}

ЛОГ ЭКСПЕРИМЕНТОВ:
{experiment_log}

Напиши отчёт в markdown формате:
1. Итоговый результат (улучшение/ухудшение метрики)
2. Что сработало (кратко)
3. Что не сработало (кратко)
4. Финальный набор изменений"""


def build_final_report_prompt(state: dict[str, Any], baseline_cv: float, final_cv: float) -> str:
    exp_log = state.get("experiment_log", [])
    kept = [e for e in exp_log if e.get("decision") == "keep"]
    rejected = [e for e in exp_log if e.get("decision") == "reject"]
    log_text, _ = _format_experiment_log(exp_log, last_n=50)
    return FINAL_REPORT_PROMPT.format(
        task=state.get("task", "?"),
        metric_name=state.get("metric_name", "?"),
        target=state.get("target", "?"),
        goal=state.get("goal", ""),
        baseline_cv=baseline_cv,
        final_cv=final_cv,
        n_iterations=len(exp_log),
        n_kept=len(kept),
        n_rejected=len(rejected),
        experiment_log=log_text,
    )
