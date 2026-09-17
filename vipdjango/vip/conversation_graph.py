from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


TARGET_TURNS = 25
# ConversationEngine emits a character-side pause on the final allowed turn.
MAX_TURNS = 25
_STAGE_ORDER = {"beginning": 0, "middle": 1, "ending": 2}


class ConversationState(TypedDict, total=False):
    scenario: dict[str, Any]
    history: list[dict[str, str]]
    active_objective: str
    covered_objectives: list[str]
    unresolved_objectives: list[str]
    recent_topics: list[str]
    active_topic: str
    covered_topics: list[str]
    unresolved_topics: list[str]
    topic_turn_counts: dict[str, int]
    topic_progress_ready: bool
    ending_ready: bool
    current_turn: int
    target_turns: int
    max_turns: int
    turns_remaining: int
    current_stage: str
    stage_transition_ready: bool
    phase: str
    response: str
    voice_metadata: str
    completion_status: bool
    complete: bool
    interrupt_requested: bool
    debug_info: dict[str, Any]


ResponseFn = Callable[[ConversationState], tuple[str, str, str, bool, bool, dict[str, Any]]]


def _objective_ids(scenario):
    return [
        objective.get("id", "").strip()
        for objective in (scenario or {}).get("objectives", [])
        if isinstance(objective, dict) and objective.get("id", "").strip()
    ]


def _valid_objective_ids(values, valid_ids):
    return list(dict.fromkeys(value for value in (values or []) if value in valid_ids))


def _phase_for_turn(turn: int) -> str:
    if turn <= 10:
        return "normal"
    if turn <= 17:
        return "resolution_guidance"
    if turn <= TARGET_TURNS + 2:
        return "closure_preference"
    return "closure_flexibility"


def build_conversation_graph(response_fn: ResponseFn):
    """Create the small state/lifecycle graph; response_fn owns natural language."""

    def initialize(state: ConversationState) -> ConversationState:
        max_turns = state.get("max_turns", MAX_TURNS)
        target_turns = state.get("target_turns", TARGET_TURNS)
        current_turn = state.get("current_turn", 0)
        if current_turn > max_turns:
            raise ValueError("Conversation turn limit exceeded.")
        valid_objectives = _objective_ids(state.get("scenario"))
        covered_objectives = _valid_objective_ids(state.get("covered_objectives"), valid_objectives)
        unresolved_objectives = _valid_objective_ids(state.get("unresolved_objectives"), valid_objectives)
        if valid_objectives and not covered_objectives and not unresolved_objectives:
            unresolved_objectives = list(valid_objectives)
        unresolved_objectives = [
            objective_id for objective_id in unresolved_objectives if objective_id not in covered_objectives
        ]
        active_objective = state.get("active_objective", "")
        if active_objective not in unresolved_objectives:
            active_objective = unresolved_objectives[0] if unresolved_objectives else ""
        return {
            **state,
            "active_objective": active_objective,
            "covered_objectives": covered_objectives,
            "unresolved_objectives": unresolved_objectives,
            "recent_topics": list(state.get("recent_topics", [])),
            "ending_ready": bool(state.get("ending_ready", False)),
            "target_turns": target_turns,
            "max_turns": max_turns,
            "turns_remaining": max(0, max_turns - current_turn),
            "current_stage": state.get("current_stage", "beginning"),
            "stage_transition_ready": state.get("stage_transition_ready", False),
            "phase": _phase_for_turn(current_turn),
            "completion_status": state.get("completion_status", False),
            "interrupt_requested": state.get("interrupt_requested", False),
        }

    def generate(state: ConversationState) -> ConversationState:
        result = response_fn(state)
        if len(result) == 3:
            response, complete, debug_info = result
            voice = ""
            stage = debug_info.get("stage", state.get("current_stage", "beginning"))
            interrupted = debug_info.get("interrupt_requested", False)
        else:
            response, voice, stage, complete, interrupted, debug_info = result
        transition_ready = bool(debug_info.get("stage_transition_ready", False))
        current_stage = state.get("current_stage", "beginning")
        if stage not in _STAGE_ORDER or _STAGE_ORDER[stage] < _STAGE_ORDER[current_stage]:
            stage = current_stage
            transition_ready = False
        elif stage != current_stage:
            # A forward stage selected by the response adapter is the
            # semantic transition signal. Do not discard it because a second,
            # redundant readiness flag was false or omitted.
            transition_ready = True
        updated_state = {
            **state,
            "response": response,
            "voice_metadata": voice,
            "current_stage": stage or state["current_stage"],
            "stage_transition_ready": transition_ready,
            "completion_status": complete,
            "complete": complete,
            "interrupt_requested": interrupted,
            "debug_info": debug_info,
        }
        for field in (
            "active_objective",
            "covered_objectives",
            "unresolved_objectives",
            "recent_topics",
            "active_topic",
            "covered_topics",
            "unresolved_topics",
            "topic_turn_counts",
            "topic_progress_ready",
            "ending_ready",
        ):
            if field in debug_info:
                updated_state[field] = debug_info[field]
        return updated_state

    def finalize(state: ConversationState) -> ConversationState:
        # Completion is semantic. The graph must not turn either the soft
        # target or the operational safety boundary into a normal closing
        # transition.
        complete = state.get("completion_status", False)
        return {
            **state,
            "completion_status": complete,
            "complete": complete,
            "turns_remaining": max(0, state["max_turns"] - state["current_turn"]),
        }

    graph = StateGraph(ConversationState)
    graph.add_node("initialize", initialize)
    graph.add_node("generate", generate)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "generate")
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
