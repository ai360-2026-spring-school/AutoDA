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

=== ТИПЫ КОЛОНОК ===
{column_type_map}

=== ТВОЯ ПАМЯТЬ ===
{planner_memory}

=== ЛОГ ЭКСПЕРИМЕНТОВ (последние {log_count}) ===
{experiment_log}

=== ТЕКУЩАЯ ИТЕРАЦИЯ (шаг {step}) — уже применено ===
{current_iter_transforms}

=== СООБЩЕНИЕ КРИТИКА ===
{critic_message}

=== КАТАЛОГ ОПЕРАЦИЙ ===
{catalog}

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
- Не повторяй уже отклонённые операции из лога
- group_aggregate принимает ОДИН by (str) и ОДИН value (str), не список
- multi_col_lambda: для колонок с пробелами/спецсимволами ОБЯЗАТЕЛЬНО используй col('имя').
  Пример: {{"expression": "col('LapTime (s)') * TyreLife", "input_columns": ["LapTime (s)", "TyreLife"], "result_column": "lap_x_tyre"}}
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
# Critic prompt
# ---------------------------------------------------------------------------

CRITIC_PROMPT = """\
Ты — критик агента по разработке признаков.

ОПИСАНИЕ ЗАДАЧИ:
{long_summary}

ЛОГ ВСЕХ ЭКСПЕРИМЕНТОВ (последние {log_count}):
{experiment_log}

Твоя задача: проанализировать поведение агента и дать короткий комментарий (1-3 предложения) ТОЛЬКО если видишь реальную проблему:
- Агент несколько раз подряд пробует похожие изменения, которые отклоняются
- Агент игнорирует очевидно важные колонки из описания задачи
- Агент застрял в бесполезном цикле действий

Если поведение агента разумное — верни null.

Ответь СТРОГО в формате JSON: {{"message": "текст критики" или null}}"""


def build_critic_prompt(state: dict[str, Any]) -> str:
    long_summary = (state.get("long_description_summary") or "(описание не задано)")[:1000]
    exp_log = state.get("experiment_log", [])
    log_text, log_count = _format_experiment_log(exp_log, last_n=20)
    return CRITIC_PROMPT.format(
        long_summary=long_summary,
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
