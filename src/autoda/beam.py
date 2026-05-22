"""
BeamSearchAgent — tree / beam search over feature engineering space.

Architecture:
  Root (df₀, cv₀)
  → Planner generates N ideas  (no implementation, just text)
  → N parallel Implementers each apply one idea (clean context, no failure log)
  → Prune: keep top beam_width children by CV
  → Repeat from each survivor

Key difference from PDAgent:
  - No shared experiment_log that accumulates failures
  - Each branch is independent — clean context for each implementer
  - Parallelism at every level via ThreadPoolExecutor
  - Full search tree is recorded and can be visualised
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .actions.registry import TRANSFORMERS, catalog_text as _catalog_text
from .column_typer import detect_column_types
from .evaluator import CatBoostEvaluator, is_keep
from .graph import parse_json_response

_CATALOG_COMPACT: str = _catalog_text()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    df: pd.DataFrame
    test_df: pd.DataFrame | None
    cv: float
    idea_used: str | None          # text idea that created this node
    transforms_chain: list[dict]   # all transforms applied since root
    transforms_in_step: list[dict] # transforms applied in THIS step only
    children_ids: list[str] = field(default_factory=list)
    pruned: bool = False
    error: str | None = None
    critic_directive: str | None = None   # set after pruning, used in next ideate

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def branch_summary(self) -> str:
        """One-line summary of this branch for the implementer prompt."""
        if not self.transforms_chain:
            return "(исходный датасет, трансформации не применялись)"
        ops = [f'{t["op"]}({json.dumps(t.get("args", {}), ensure_ascii=False)[:40]})'
               for t in self.transforms_chain]
        return " → ".join(ops)

    def new_columns(self, root_columns: set[str]) -> list[str]:
        """Columns added in this branch vs root."""
        return [c for c in self.df.columns if c not in root_columns]


@dataclass
class BeamSearchResult:
    best_node: TreeNode
    all_nodes: dict[str, TreeNode]
    root_cv: float
    best_cv: float
    reports_dir: Path

    @property
    def improvement_pct(self) -> float:
        if self.root_cv == 0:
            return 0.0
        return abs(self.root_cv - self.best_cv) / abs(self.root_cv) * 100

    @property
    def final_df(self) -> pd.DataFrame:
        return self.best_node.df

    @property
    def final_test_df(self) -> pd.DataFrame | None:
        return self.best_node.test_df


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_IDEA_GENERATOR_PROMPT = """\
Ты — генератор идей для feature engineering. НЕ реализуй ничего — только генерируй идеи.

ЗАДАЧА: {task} | МЕТРИКА: {metric_name} ({metric_direction}) | ТАРГЕТ: {target}
ЦЕЛЬ: {goal}
{description_line}

ТЕКУЩИЕ КОЛОНКИ ДАТАСЕТА:
{columns}

ТЕКУЩИЙ CV: {current_cv:.4f} (baseline: {root_cv:.4f})
УЖЕ ПРИМЕНЕНО В ЭТОЙ ВЕТКЕ:
{branch_summary}
НОВЫЕ КОЛОНКИ ДОБАВЛЕННЫЕ В ЭТОЙ ВЕТКЕ (используй их в идеях!):
{new_columns}

ЧТО ПОЛЕЗНО ДЛЯ CATBOOST (деревья решений):
  ✓ Произведения двух непрерывных: A × B
  ✓ Отношения: A / B
  ✓ Разности: A − B
  ✓ Логарифм сильно скошенных фич
  ✓ Агрегаты по категории (mean_target_by_group)
  ✗ Нормализация/стандартизация — деревьям не нужна
  ✗ Полиномы (x²) — деревья строят сами через сплиты

╔══════════════════════════════════════════════════════════╗
║  ДИРЕКТИВА КРИТИКА — ОБЯЗАТЕЛЬНО ВКЛЮЧИ КАК ПЕРВУЮ ИДЕЮ ║
╚══════════════════════════════════════════════════════════╝
{critic_directive}

Сгенерируй РОВНО {n_ideas} разных идей. Первая идея — ОБЯЗАТЕЛЬНО директива критика выше (если есть).
Остальные — разнообразные идеи с реальными именами колонок из раздела "ТЕКУЩИЕ КОЛОНКИ" выше.
ЗАПРЕЩЕНО использовать в идее колонку которой нет в разделе "ТЕКУЩИЕ КОЛОНКИ"!

Ответь СТРОГО JSON списком без пояснений:
[
  {{"idea": "добавить interaction(cement, water, op=div)", "rationale": "cement/water ratio — главный предиктор прочности"}},
  {{"idea": "log_transform(age, plus_one=true)", "rationale": "возраст имеет нелинейный эффект"}},
  ...
]"""


_CRITIC_PROMPT = """\
Ты — критик feature engineering для одной конкретной ветки поиска.

ЗАДАЧА: {task} | МЕТРИКА: {metric_name} ({metric_direction}) | ТАРГЕТ: {target}
ЦЕЛЬ: {goal}
{description_line}

ТЕКУЩИЕ КОЛОНКИ ЭТОЙ ВЕТКИ (ТОЛЬКО эти существуют):
{columns}

НОВЫЕ КОЛОНКИ ДОБАВЛЕННЫЕ В ЭТОЙ ВЕТКЕ:
{new_columns}

ТЕКУЩИЙ CV ВЕТКИ: {current_cv:.4f} (baseline: {root_cv:.4f})
ПУТЬ ВЕТКИ (что применено):
{branch_summary}

ЧТО РАБОТАЕТ ДЛЯ CATBOOST: произведения (A×B), отношения (A/B), разности (A−B), log скошенных, group_aggregate.
ЧТО НЕ РАБОТАЕТ: нормализация, полиномы, стандартизация.

Твоя задача: дать ОДНУ конкретную директиву — что попробовать следующим из ЭТОЙ ветки.
Используй ТОЛЬКО колонки из списка "ТЕКУЩИЕ КОЛОНКИ ЭТОЙ ВЕТКИ" выше.
Если ничего очевидного нет — верни null.

Ответь СТРОГО JSON: {{"directive": "конкретная операция с именами колонок" или null}}"""


_IMPLEMENTER_PROMPT = """\
Ты — реализатор feature engineering. Твоя задача: реализовать ОДНУ конкретную идею.

ЗАДАЧА: {task} | МЕТРИКА: {metric_name} ({metric_direction}) | ТАРГЕТ: {target}
{description_line}

ТЕКУЩИЕ КОЛОНКИ ДАТАСЕТА:
{columns}

ТЕКУЩИЙ CV: {current_cv:.4f}
ИСТОРИЯ ВЕТКИ (что уже применено):
{branch_summary}

━━━ ТВОЯ ЗАДАЧА НА ЭТОМ ШАГЕ ━━━
{idea}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

КАТАЛОГ ОПЕРАЦИЙ:
{catalog}

ПРАВИЛА:
- Реализуй ТОЛЬКО идею выше, не добавляй ничего лишнего
- Используй ТОЛЬКО колонки из списка выше — не используй колонки которых там нет
- Делай не более 2 трансформаций, затем submit
- multi_col_lambda: для имён с пробелами/спецсимволами используй col('имя')
- multi_col_lambda ОБЯЗАТЕЛЬНО принимает input_columns — список всех колонок из expression
- group_aggregate: один by (str) и один value (str)

Доступные действия (строгий JSON):
{{"type": "transform", "op": "<op>", "args": {{...}}}}
{{"type": "submit", "rationale": "обоснование"}}
{{"type": "cancel"}}"""

_IMPLEMENTER_ADDENDUM = {
    "transform_applied": "\n=== ПРИМЕНЕНО: {op}({args}) — новые колонки: {new_cols} ===\nПрименено в шаге: {all_transforms}\n",
    "transform_error": "\n=== ОШИБКА: {op}({args}) — {error} ===\nПопробуй другой подход или submit без изменений.\n",
    "cancelled": "\n=== ОТМЕНЕНО ===\nПопробуй submit без изменений.\n",
    "parse_error": "\n=== ОШИБКА ПАРСИНГА: верни строгий JSON ===\n",
}


def _build_idea_generator_prompt(
    node: TreeNode,
    target: str,
    task: str,
    metric_name: str,
    metric_direction: str,
    root_cv: float,
    goal: str,
    description: str | None,
    n_ideas: int,
    root_columns: set[str],
) -> str:
    col_map = detect_column_types(node.df, target)
    cols_str = "\n".join(
        f"  {col} ({kind})" for col, kind in sorted(col_map.items()) if col != target
    ) or "  (нет данных)"
    desc_line = f"\nОПИСАНИЕ: {description[:300]}" if description else ""
    new_cols = node.new_columns(root_columns)
    new_cols_str = ", ".join(new_cols) if new_cols else "(нет — это корневой узел)"
    critic_dir = node.critic_directive or "(нет директивы)"
    return _IDEA_GENERATOR_PROMPT.format(
        task=task, metric_name=metric_name, metric_direction=metric_direction,
        target=target, goal=goal, description_line=desc_line,
        columns=cols_str, current_cv=node.cv, root_cv=root_cv,
        branch_summary=node.branch_summary(), new_columns=new_cols_str,
        critic_directive=critic_dir, n_ideas=n_ideas,
    )


def _build_critic_prompt(
    node: TreeNode,
    target: str,
    task: str,
    metric_name: str,
    metric_direction: str,
    root_cv: float,
    goal: str,
    description: str | None,
    root_columns: set[str],
) -> str:
    col_map = detect_column_types(node.df, target)
    cols_str = "\n".join(
        f"  {col} ({kind})" for col, kind in sorted(col_map.items()) if col != target
    ) or "  (нет данных)"
    desc_line = f"\nОПИСАНИЕ: {description[:300]}" if description else ""
    new_cols = node.new_columns(root_columns)
    new_cols_str = ", ".join(new_cols) if new_cols else "(нет)"
    return _CRITIC_PROMPT.format(
        task=task, metric_name=metric_name, metric_direction=metric_direction,
        target=target, goal=goal, description_line=desc_line,
        columns=cols_str, current_cv=node.cv, root_cv=root_cv,
        branch_summary=node.branch_summary(), new_columns=new_cols_str,
    )


def _build_implementer_prompt(
    node: TreeNode,
    target: str,
    task: str,
    metric_name: str,
    metric_direction: str,
    idea: str,
    description: str | None,
    addendum: str = "",
) -> str:
    col_map = detect_column_types(node.df, target)
    cols_str = "\n".join(
        f"  {col} ({kind})" for col, kind in sorted(col_map.items()) if col != target
    ) or "  (нет данных)"
    desc_line = f"\nОПИСАНИЕ: {description[:200]}" if description else ""
    base = _IMPLEMENTER_PROMPT.format(
        task=task, metric_name=metric_name, metric_direction=metric_direction,
        target=target, description_line=desc_line,
        columns=cols_str, current_cv=node.cv,
        branch_summary=node.branch_summary(),
        idea=idea, catalog=_CATALOG_COMPACT,
    )
    return (base + addendum).strip()


# ---------------------------------------------------------------------------
# BeamSearchAgent
# ---------------------------------------------------------------------------

class BeamSearchAgent:
    def __init__(
        self,
        planner_model,
        implementer_model=None,
        critic_model=None,
        evaluator: CatBoostEvaluator | None = None,
        *,
        beam_width: int = 2,
        n_ideas: int = 4,
        max_depth: int = 5,
        max_impl_turns: int = 5,
        tolerance: float = 1e-4,
        debug: bool = False,
    ):
        self.planner = planner_model
        self.implementer = implementer_model if implementer_model is not None else planner_model
        self.critic = critic_model  # None = no critic
        self.evaluator = evaluator
        self.beam_width = beam_width
        self.n_ideas = n_ideas
        self.max_depth = max_depth
        self.max_impl_turns = max_impl_turns
        self.tolerance = tolerance
        self.debug = debug

    def _dbg(self, *parts: Any) -> None:
        if self.debug:
            print(" ".join(str(p) for p in parts), flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        target: str,
        goal: str,
        task: str,
        metric_name: str,
        metric_direction: str,
        test_df: pd.DataFrame | None = None,
        description: str | None = None,
        reports_dir: Path = Path("reports/beam"),
        numeric_unique_threshold: int = 12,
    ) -> BeamSearchResult:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        self._target = target
        self._task = task
        self._metric_name = metric_name
        self._metric_direction = metric_direction
        self._goal = goal
        self._description = description
        self._numeric_unique_threshold = numeric_unique_threshold

        # Baseline CV
        X0 = df.drop(columns=[target], errors="ignore")
        y0 = df[target]
        baseline = self.evaluator.cv(X0, y0, step=0)
        root_cv = baseline.mean

        root = TreeNode(
            node_id="root",
            parent_id=None,
            depth=0,
            df=df.copy(),
            test_df=test_df.copy() if test_df is not None else None,
            cv=root_cv,
            idea_used=None,
            transforms_chain=[],
            transforms_in_step=[],
        )
        self._root_columns: set[str] = set(df.columns)
        self._root_cv: float = root_cv

        all_nodes: dict[str, TreeNode] = {"root": root}
        beam: list[TreeNode] = [root]

        self._dbg(f"[beam] root cv={root_cv:.4f} | beam_width={self.beam_width} n_ideas={self.n_ideas} max_depth={self.max_depth}")

        for depth in range(1, self.max_depth + 1):
            self._dbg(f"[beam] === depth {depth} | beam size={len(beam)} ===")
            next_candidates: list[TreeNode] = []

            # For each node in current beam, generate ideas + implement in parallel
            for parent_node in beam:
                ideas = self._generate_ideas(parent_node, root_cv)
                if not ideas:
                    self._dbg(f"[beam] {parent_node.node_id}: no ideas generated")
                    continue

                children = self._implement_parallel(parent_node, ideas, depth, all_nodes)
                for child in children:
                    parent_node.children_ids.append(child.node_id)
                    all_nodes[child.node_id] = child
                    next_candidates.append(child)

            if not next_candidates:
                self._dbg(f"[beam] no candidates at depth {depth}, stopping")
                break

            # Sort all candidates by CV (best first)
            sign = 1 if metric_direction == "max" else -1
            next_candidates.sort(key=lambda n: sign * n.cv, reverse=True)

            # Prune: mark all but top beam_width as pruned
            survivors = next_candidates[:self.beam_width]
            for node in next_candidates[self.beam_width:]:
                node.pruned = True

            surviving_cvs = [f"{n.cv:.4f}" for n in survivors]
            self._dbg(f"[beam] depth {depth}: {len(survivors)}/{len(next_candidates)} survive — cvs={surviving_cvs}")

            # Run critic for each surviving branch (parallel)
            if self.critic is not None:
                with ThreadPoolExecutor(max_workers=len(survivors)) as pool:
                    critic_futures = {
                        pool.submit(self._run_critic, node): node
                        for node in survivors
                    }
                    for future in as_completed(critic_futures):
                        node = critic_futures[future]
                        directive = future.result()
                        node.critic_directive = directive
                        if directive:
                            _dbg_short = directive[:70]
                            self._dbg(f"[critic/{node.node_id}] {_dbg_short!r}")

            # Check if any survivor actually improved
            best_cv = survivors[0].cv
            improved = is_keep(
                type("R", (), {"mean": best_cv, "metric_direction": metric_direction})(),
                type("R", (), {"mean": root_cv, "metric_direction": metric_direction})(),
                tol=self.tolerance,
            )
            if not improved:
                self._dbg(f"[beam] no improvement at depth {depth}, stopping")
                break

            beam = survivors

        # Find overall best node
        sign = 1 if metric_direction == "max" else -1
        best_node = max(all_nodes.values(), key=lambda n: sign * n.cv)

        self._dbg(f"[beam] done | root={root_cv:.4f} best={best_node.cv:.4f} ({best_node.node_id})")

        # Save tree JSON
        try:
            tree_data = {
                nid: {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "cv": n.cv,
                    "idea_used": n.idea_used,
                    "transforms_in_step": n.transforms_in_step,
                    "children_ids": n.children_ids,
                    "pruned": n.pruned,
                    "error": n.error,
                }
                for nid, n in all_nodes.items()
            }
            (reports_dir / "search_tree.json").write_text(
                json.dumps(tree_data, indent=2, ensure_ascii=False)
            )
        except Exception:
            pass

        return BeamSearchResult(
            best_node=best_node,
            all_nodes=all_nodes,
            root_cv=root_cv,
            best_cv=best_node.cv,
            reports_dir=reports_dir,
        )

    # ------------------------------------------------------------------
    # Idea generation
    # ------------------------------------------------------------------

    def _generate_ideas(self, node: TreeNode, root_cv: float) -> list[str]:
        prompt = _build_idea_generator_prompt(
            node=node,
            target=self._target,
            task=self._task,
            metric_name=self._metric_name,
            metric_direction=self._metric_direction,
            root_cv=root_cv,
            goal=self._goal,
            description=self._description,
            n_ideas=self.n_ideas,
            root_columns=self._root_columns,
        )
        try:
            response = self.planner.invoke(prompt)
            content = getattr(response, "content", None) or ""
            # Parse JSON list
            content = content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            ideas_raw = json.loads(content)
            if not isinstance(ideas_raw, list):
                ideas_raw = [ideas_raw]
            ideas = []
            for item in ideas_raw:
                if isinstance(item, dict):
                    idea_text = item.get("idea", "")
                    rationale = item.get("rationale", "")
                    if idea_text:
                        ideas.append(f"{idea_text} ({rationale})" if rationale else idea_text)
                elif isinstance(item, str):
                    ideas.append(item)
            self._dbg(f"[beam/{node.node_id}] generated {len(ideas)} ideas")
            return ideas
        except Exception as e:
            self._dbg(f"[beam/{node.node_id}] idea generation failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def _run_critic(self, node: TreeNode) -> str | None:
        """Run critic for one branch. Returns directive string or None."""
        if self.critic is None:
            return None
        prompt = _build_critic_prompt(
            node=node,
            target=self._target,
            task=self._task,
            metric_name=self._metric_name,
            metric_direction=self._metric_direction,
            root_cv=self._root_cv,
            goal=self._goal,
            description=self._description,
            root_columns=self._root_columns,
        )
        try:
            response = self.critic.invoke(prompt)
            content = getattr(response, "content", None) or ""
            parsed = parse_json_response(content)
            if "_parse_error" in parsed:
                return None
            directive = parsed.get("directive")
            return directive if directive and directive != "null" else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Parallel implementation
    # ------------------------------------------------------------------

    def _implement_parallel(
        self,
        parent: TreeNode,
        ideas: list[str],
        depth: int,
        all_nodes: dict[str, TreeNode],
    ) -> list[TreeNode]:
        workers = min(len(ideas), 6)
        children: list[TreeNode] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._implement_one, parent, idea, depth, i): idea
                for i, idea in enumerate(ideas)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    children.append(result)
        return children

    def _implement_one(
        self, parent: TreeNode, idea: str, depth: int, idx: int
    ) -> TreeNode | None:
        node_id = f"d{depth}_i{idx}_{str(uuid.uuid4())[:6]}"
        self._dbg(f"[impl/{node_id}] idea: {idea[:60]}")

        # Working copies
        working_df = parent.df.copy()
        working_test_df = parent.test_df.copy() if parent.test_df is not None else None
        transforms_in_step: list[dict] = []
        addendum = ""

        for turn in range(self.max_impl_turns):
            prompt = _build_implementer_prompt(
                node=TreeNode(
                    node_id=node_id, parent_id=parent.node_id, depth=depth,
                    df=working_df, test_df=working_test_df,
                    cv=parent.cv, idea_used=idea,
                    transforms_chain=parent.transforms_chain + transforms_in_step,
                    transforms_in_step=transforms_in_step,
                ),
                target=self._target,
                task=self._task,
                metric_name=self._metric_name,
                metric_direction=self._metric_direction,
                idea=idea,
                description=self._description,
                addendum=addendum,
            )

            try:
                response = self.implementer.invoke(prompt)
                text = getattr(response, "content", None) or ""
            except Exception as e:
                self._dbg(f"[impl/{node_id}] LLM error: {e}")
                addendum = _IMPLEMENTER_ADDENDUM["parse_error"]
                continue

            action = parse_json_response(text)
            if "_parse_error" in action:
                self._dbg(f"[impl/{node_id}] parse error turn {turn+1}")
                addendum = _IMPLEMENTER_ADDENDUM["parse_error"]
                continue

            action_type = action.get("type", "")

            if action_type == "submit":
                break

            if action_type == "cancel":
                working_df = parent.df.copy()
                working_test_df = parent.test_df.copy() if parent.test_df is not None else None
                transforms_in_step = []
                addendum = _IMPLEMENTER_ADDENDUM["cancelled"]
                break

            if action_type == "transform":
                op = action.get("op", "")
                args = action.get("args", {})
                if op not in TRANSFORMERS:
                    addendum = _IMPLEMENTER_ADDENDUM["transform_error"].format(
                        op=op, args=args, error=f"unknown op {op!r}"
                    )
                    continue
                try:
                    df_out, transformer, _ = TRANSFORMERS[op](working_df, self._target, **args)
                    new_cols = [c for c in df_out.columns if c not in working_df.columns]
                    working_df = df_out
                    if working_test_df is not None:
                        try:
                            working_test_df = transformer.apply(working_test_df)
                        except Exception:
                            pass
                    transforms_in_step.append({"op": op, "args": args})
                    addendum = _IMPLEMENTER_ADDENDUM["transform_applied"].format(
                        op=op,
                        args=json.dumps(args, ensure_ascii=False)[:60],
                        new_cols=new_cols,
                        all_transforms=transforms_in_step,
                    )
                except Exception as e:
                    self._dbg(f"[impl/{node_id}] transform error: {e}")
                    addendum = _IMPLEMENTER_ADDENDUM["transform_error"].format(
                        op=op, args=json.dumps(args)[:60], error=str(e)[:80]
                    )
                continue

            # stop or unknown → submit
            break

        # Evaluate
        if not transforms_in_step:
            self._dbg(f"[impl/{node_id}] no transforms applied, skipping")
            return None

        try:
            X = working_df.drop(columns=[self._target], errors="ignore")
            y = working_df[self._target]
            result = self.evaluator.cv(X, y, step=None)
            new_cv = result.mean
        except Exception as e:
            self._dbg(f"[impl/{node_id}] CV error: {e}")
            return TreeNode(
                node_id=node_id, parent_id=parent.node_id, depth=depth,
                df=working_df, test_df=working_test_df, cv=parent.cv,
                idea_used=idea,
                transforms_chain=parent.transforms_chain + transforms_in_step,
                transforms_in_step=transforms_in_step,
                error=str(e),
            )

        sign = 1 if self._metric_direction == "max" else -1
        delta = sign * (new_cv - parent.cv)
        self._dbg(f"[impl/{node_id}] cv={new_cv:.4f} delta={delta:+.4f} ops={[t['op'] for t in transforms_in_step]}")

        return TreeNode(
            node_id=node_id, parent_id=parent.node_id, depth=depth,
            df=working_df, test_df=working_test_df, cv=new_cv,
            idea_used=idea,
            transforms_chain=parent.transforms_chain + transforms_in_step,
            transforms_in_step=transforms_in_step,
        )
