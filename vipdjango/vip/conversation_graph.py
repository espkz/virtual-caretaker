from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


TARGET_TURNS = 20
# A safety cap is still useful, but it is deliberately separate from the
# approximate target. Reaching TARGET_TURNS must never complete a scenario.
MAX_TURNS = TARGET_TURNS + 4


class ConversationState(TypedDict, total=False):
    role_text: str
    scenario: dict[str, Any]
    history: list[dict[str, str]]
    current_turn: int
    target_turns: int
    max_turns: int
    turns_remaining: int
    hard_limit_reached: bool
    current_stage: str
    stage_transition_ready: bool
    phase: str
    conversation_stage: str
    response: str
    voice_metadata: str
    completion_status: bool
    complete: bool
    interrupt_requested: bool
    introduction_emitted: bool
    protocol_emitted: bool
    debug_info: dict[str, Any]


ResponseFn = Callable[[ConversationState], tuple[str, str, str, bool, bool, dict[str, Any]]]


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
        return {
            **state,
            "target_turns": target_turns,
            "max_turns": max_turns,
            "turns_remaining": max(0, max_turns - current_turn),
            "hard_limit_reached": current_turn >= max_turns,
            "current_stage": state.get("current_stage", "beginning"),
            "stage_transition_ready": state.get("stage_transition_ready", False),
            "phase": _phase_for_turn(current_turn),
            "conversation_stage": state.get("conversation_stage", "normal"),
            "completion_status": state.get("completion_status", False),
            "interrupt_requested": state.get("interrupt_requested", False),
            "introduction_emitted": state.get("introduction_emitted", False),
            "protocol_emitted": state.get("protocol_emitted", False),
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
        if stage != state.get("current_stage", "beginning") and not transition_ready:
            stage = state.get("current_stage", "beginning")
        return {
            **state,
            "response": response,
            "voice_metadata": voice,
            "current_stage": stage or state["current_stage"],
            "conversation_stage": stage or state["current_stage"],
            "stage_transition_ready": transition_ready,
            "completion_status": complete,
            "complete": complete,
            "interrupt_requested": interrupted,
            "debug_info": debug_info,
        }

    def finalize(state: ConversationState) -> ConversationState:
        # Completion is semantic. The response adapter may use the hard cap
        # as an emergency fallback, but the graph must not turn a numeric
        # target into a normal closing transition.
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
