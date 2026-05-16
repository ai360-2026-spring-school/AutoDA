import json
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .dataset import DatasetContext, profile_dataframe
from .tools import PandasTools
from .prompts import PLANNER_PROMPT, REFLECT_PROMPT, FINAL_REPORT_PROMPT


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "error": f"Could not parse JSON: {e}",
            "raw": text,
        }


def build_graph(model, context: DatasetContext, max_iterations: int = 8):
    tools = PandasTools(context.df, target=context.target)

    def profile_node(state: AgentState) -> dict[str, Any]:
        return {
            "dataset_profile": profile_dataframe(context.df),
            "current_step": 0,
            "iterations": [],
            "decision": "continue",
            "last_error": None,
            "final_report": None,
        }

    def planner_node(state: AgentState) -> dict[str, Any]:
        msg = f"""
                {PLANNER_PROMPT}

                Goal:
                {state["goal"]}

                Target:
                {state.get("target")}

                Dataset profile:
                {json.dumps(state["dataset_profile"], ensure_ascii=False)[:12000]}

                Previous iterations:
                {json.dumps(state["iterations"], ensure_ascii=False)[:12000]}
                """

        response = model.invoke(msg)
        content = getattr(response, "content", str(response))
        action = parse_json_response(content)

        if "error" in action:
            return {
                "last_error": action["error"],
                "proposed_action": None,
            }

        return {
            "proposed_action": action,
            "last_error": None,
        }

    def execute_node(state: AgentState) -> dict[str, Any]:
        action = state.get("proposed_action")

        if not action:
            return {
                "last_observation": None,
                "last_error": "No proposed action",
            }

        operation = action.get("operation")
        args = action.get("args", {})

        try:
            if operation == "describe_columns":
                observation = tools.describe_columns(**args)
            elif operation == "groupby_agg":
                observation = tools.groupby_agg(**args)
            elif operation == "correlation_with_target":
                observation = tools.correlation_with_target()
            else:
                observation = {"error": f"Unsupported operation: {operation}"}

            return {
                "last_observation": observation,
                "last_error": observation.get("error"),
            }

        except Exception as e:
            return {
                "last_observation": None,
                "last_error": repr(e),
            }

    def reflect_node(state: AgentState) -> dict[str, Any]:
        msg = f"""
{REFLECT_PROMPT}

Goal:
{state["goal"]}

Action:
{json.dumps(state["proposed_action"], ensure_ascii=False)}

Observation:
{json.dumps(state["last_observation"], ensure_ascii=False)[:12000]}

Error:
{state.get("last_error")}
"""

        response = model.invoke(msg)
        content = getattr(response, "content", str(response))
        reflection = parse_json_response(content)

        conclusion = reflection.get("conclusion", "")
        decision = reflection.get("decision", "continue")

        step = state.get("current_step", 0) + 1

        iteration = {
            "step": step,
            "thought": state["proposed_action"].get("thought", ""),
            "action": state["proposed_action"],
            "observation": state.get("last_observation") or {},
            "conclusion": conclusion,
        }

        if step >= max_iterations:
            decision = "finish"

        return {
            "iterations": [iteration],
            "current_step": step,
            "decision": decision,
        }

    def final_report_node(state: AgentState) -> dict[str, Any]:
        msg = f"""
{FINAL_REPORT_PROMPT}

Goal:
{state["goal"]}

Iterations:
{json.dumps(state["iterations"], ensure_ascii=False)[:20000]}
"""

        response = model.invoke(msg)
        content = getattr(response, "content", str(response))

        return {
            "final_report": content,
        }

    def route_after_reflect(state: AgentState) -> Literal["planner", "final_report"]:
        if state.get("decision") == "finish":
            return "final_report"
        return "planner"

    graph = StateGraph(AgentState)

    graph.add_node("profile", profile_node)
    graph.add_node("planner", planner_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "planner")
    graph.add_edge("planner", "execute")
    graph.add_edge("execute", "reflect")

    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "planner": "planner",
            "final_report": "final_report",
        },
    )

    graph.add_edge("final_report", END)

    return graph.compile()
