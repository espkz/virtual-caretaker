import json
import logging
import re

from .conversation_graph import MAX_TURNS, TARGET_TURNS, build_conversation_graph
from .conversation_scenario import Scenario, parse_scenario_prompt


logger = logging.getLogger(__name__)


_EMBEDDED_VOICE_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*$", flags=re.DOTALL)


def split_dialogue_and_voice(text):
    """Separate legacy trailing voice annotations from spoken dialogue.

    New responses carry these values in separate structured fields. This
    helper exists at the boundary for old database rows and for a model that
    violates the output contract by putting an annotation in ``dialogue``.
    It only recognizes a final bracketed annotation; learner text is never
    passed through this function as dialogue.
    """
    value = (text or "").strip()
    match = _EMBEDDED_VOICE_RE.search(value)
    if not match:
        return value, ""
    dialogue = value[: match.start()].rstrip()
    return dialogue, match.group(1).strip()


def _history(messages):
    """Convert the complete persisted transcript to native LLM roles."""
    result = []
    for message in messages:
        sender = str(message.sender)
        if sender not in {"student", "assistant"}:
            continue
        content = (message.content or "").strip()
        if sender == "assistant":
            content, _ = split_dialogue_and_voice(content)
        result.append({
            "role": "user" if sender == "student" else "assistant",
            "content": content,
        })
    return result


def _voice_metadata(value, scenario):
    """Normalize model voice output to ``gender voice, style`` without brackets."""
    raw = (value or "").strip().strip("[]").strip()
    default = scenario.voice_metadata()
    if not raw:
        return default

    raw = re.sub(r"^\s*(male|female)\s+voice\s*,?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^\s*(male|female)\s*,?\s*", "", raw, flags=re.IGNORECASE).strip()
    if not raw or raw.lower() in {"male", "female", "voice"}:
        return default
    return f"{scenario.voice_gender} voice, {raw}"


def format_voice_metadata(metadata):
    metadata = (metadata or "").strip().strip("[]")
    return f"[{metadata}]" if metadata else ""


class ConversationEngine:
    def __init__(self, global_prompt, api_key, model="gpt-5-nano"):
        self.global_prompt = global_prompt.strip()
        self.api_key = api_key
        self.model = model
        self.graph = build_conversation_graph(self._llm_turn)

    def _interrupt(self, message):
        from openai import OpenAI

        payload = [
            {"role": "system", "content": "Return true only when the learner clearly asks to stop or end the conversation now. Return false for ordinary roleplay, thanks with a question, or discussion of an ending."},
            {"role": "user", "content": message},
        ]
        logger.debug("Interrupt payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        response = OpenAI(api_key=self.api_key).responses.create(
            model=self.model,
            text={"format": {"type": "json_schema", "name": "interrupt", "schema": {"type": "object", "properties": {"stop_requested": {"type": "boolean"}}, "required": ["stop_requested"], "additionalProperties": False}, "strict": True}},
            input=payload,
        )
        return bool(json.loads(response.output_text or "{}").get("stop_requested"))

    def _system_prompt(self, scenario, state):
        stage = state.get("current_stage", "beginning")
        behavior = scenario[stage]
        cues = {
            "beginning": scenario["beginning_to_middle_cues"],
            "middle": scenario["middle_to_ending_cues"],
            "ending": scenario["end_of_conversation_cues"],
        }[stage]
        return (
            f"{self.global_prompt}\n\n"
            "SPEAKER CONTRACT (highest priority):\n"
            "- You are the assistant, and you are always the scenario character.\n"
            "- The user messages are the learner's actual words. The user is never you.\n"
            "- Never play, quote as new dialogue, or narrate the learner's identity, actions, thoughts, feelings, or professional decisions.\n"
            "- You may describe only what your character says, thinks, feels, knows, notices, or does.\n"
            "- Do not answer from the learner's professional perspective and do not teach the learner unless the character is asking for information.\n"
            "- Treat information supplied by the learner as information from the learner; do not silently convert it into your character's knowledge or action.\n"
            "- Never invent learner actions, thoughts, feelings, dialogue, or identity. If the learner has not said or done something, it has not happened.\n\n"
            "IMMUTABLE SCENARIO IDENTITY:\n"
            f"ASSISTANT CHARACTER (## Role):\n{scenario['character']}\n\n"
            f"LEARNER / USER (## Learner Role):\n{scenario['learner']}\n\n"
            "The assistant is the character; the user is the learner. These labels never change during the conversation.\n\n"
            "SCENARIO STAGE GUIDANCE:\n"
            f"Beginning:\n{scenario['beginning']}\n\n"
            f"Middle:\n{scenario['middle']}\n\n"
            f"Ending:\n{scenario['ending']}\n\n"
            f"CURRENT STAGE: {stage}\n"
            f"Current-stage behavioral guidance:\n{behavior}\n"
            f"Relevant semantic transition cues:\n{cues}\n"
            f"Scenario meta-instructions:\n{scenario['meta']}\n"
            f"Voice gender: {scenario['voice_gender']}\n"
            f"Baseline voice style: {scenario['voice_style']}\n"
            "Stage contract: remain in the current stage unless the relevant scenario cues have genuinely been met. "
            "Stage transitions are semantic, not turn-count thresholds. Beginning has no minimum length; move early "
            "when the learner's response has addressed the cues. Do not begin technical training while the current "
            "stage still requires unresolved character concerns.\n"
            "TURN-BUDGET CONTRACT:\n"
            f"Current learner turn: {state.get('current_turn', 0)}; soft target: {state.get('target_turns', TARGET_TURNS)}; "
            f"turns remaining before the safety cap: {state.get('turns_remaining', MAX_TURNS)}; phase: {state.get('phase', 'normal')}.\n"
            "The target is an advisory pressure against indefinite looping, not a minimum, schedule, stage rule, or "
            "termination trigger. Never complete or emit the Closing merely because a number was reached. Do not add "
            "filler questions or repeat answered topics to consume turns. Around the later budget window, prefer a "
            "natural closure only when the scenario's ending cues and the current conversational state support it; "
            "otherwise continue only as needed to address the scenario.\n"
            "VOICE/DIALOGUE CONTRACT:\n"
            "Return JSON only. The dialogue field contains only the character's spoken words. Never put voice, emotion, "
            "stage notes, brackets, narration, or metadata in dialogue. The voice field contains only a concise style "
            "description, without brackets or the word 'voice'; it is metadata for TTS and is not spoken.\n"
            "Questions are optional. Respond naturally to what the learner actually says, within the character's "
            "perspective and knowledge. Return JSON fields dialogue, voice, stage, stage_transition_ready, complete, "
            "stop_requested."
        )

    def _llm_turn(self, state):
        from openai import OpenAI

        scenario = state["scenario"]
        payload = [{"role": "system", "content": self._system_prompt(scenario, state)}, *state["history"]]
        logger.debug("LLM context stage=%s turn=%s character=%s learner=%s payload=\n%s", state["current_stage"], state["current_turn"], scenario["character"], scenario["learner"], json.dumps(payload, ensure_ascii=False, indent=2))
        response = OpenAI(api_key=self.api_key).responses.create(
            model=self.model,
            text={"format": {"type": "json_schema", "name": "character_turn", "schema": {"type": "object", "properties": {"dialogue": {"type": "string"}, "voice": {"type": "string"}, "stage": {"type": "string", "enum": ["beginning", "middle", "ending"]}, "stage_transition_ready": {"type": "boolean"}, "complete": {"type": "boolean"}, "stop_requested": {"type": "boolean"}}, "required": ["dialogue", "voice", "stage", "stage_transition_ready", "complete", "stop_requested"], "additionalProperties": False}, "strict": True}},
            input=payload,
        )
        result = json.loads(response.output_text or "{}")
        interrupted = bool(result.get("stop_requested"))
        if interrupted:
            return "", "", state["current_stage"], True, True, {"reason": "interrupt"}
        parsed_scenario = Scenario(**scenario)
        proposed_stage = result.get("stage", state["current_stage"])
        transition_ready = bool(result.get("stage_transition_ready"))
        if proposed_stage != state["current_stage"] and not transition_ready:
            proposed_stage = state["current_stage"]
        model_requested_completion = bool(result.get("complete"))
        # A completion flag is meaningful only with an explicit semantic
        # transition to Ending. This prevents the soft target from becoming
        # an implicit closing command.
        complete = model_requested_completion and proposed_stage == "ending" and transition_ready
        raw_dialogue = (result.get("dialogue", "") or "").strip()
        dialogue, embedded_voice = split_dialogue_and_voice(raw_dialogue)
        voice_value = result.get("voice", "") or embedded_voice
        if embedded_voice and (not voice_value or voice_value.strip("[] ").lower() in {"male", "female", "voice"}):
            # Preserve a useful style if a malformed response supplies only
            # the gender in the structured field but puts the style in the
            # legacy bracket position.
            voice_value = embedded_voice

        if complete and parsed_scenario.closing:
            dialogue = parsed_scenario.closing

        hard_budget_fallback = state.get("current_turn", 0) >= state.get("max_turns", MAX_TURNS) and not complete
        if hard_budget_fallback:
            dialogue = parsed_scenario.closing or dialogue
            proposed_stage = "ending"
            transition_ready = True
            complete = True

        reason = "hard_budget_fallback" if hard_budget_fallback else "llm_turn"
        return dialogue, _voice_metadata(voice_value, parsed_scenario), proposed_stage, complete, False, {
            "reason": reason,
            "stage_transition_ready": transition_ready,
            "completion_requested": model_requested_completion,
            "completion_rejected": model_requested_completion and not complete and not hard_budget_fallback,
            "embedded_voice_removed": bool(embedded_voice),
        }

    def respond(self, role_text, messages, conversation_state=None):
        scenario = parse_scenario_prompt(role_text)
        learner_messages = [m for m in messages if m.sender == "student"]
        turn = len(learner_messages)
        if turn == 0:
            return scenario.introduction, False, {"stage": "beginning", "current_stage": "beginning", "voice_metadata": scenario.voice_metadata(True), "reason": "introduction"}
        latest = learner_messages[-1].content
        try:
            if self._interrupt(latest):
                return "", True, {"stage": "ending", "current_stage": "ending", "interrupt_requested": True, "reason": "interrupt"}
        except Exception:
            logger.exception("Interrupt classification failed")
        if turn == 1 and scenario.opening_line:
            return scenario.opening_line, False, {"stage": "beginning", "current_stage": "beginning", "voice_metadata": scenario.voice_metadata(), "reason": "fixed_opening_line"}
        current_stage = (conversation_state or {}).get("current_stage")
        if current_stage not in {"beginning", "middle", "ending"}:
            current_stage = "beginning"
        state = self.graph.invoke({
            "scenario": scenario.to_state(),
            "history": _history(messages),
            "current_turn": turn,
            "target_turns": TARGET_TURNS,
            "max_turns": MAX_TURNS,
            "current_stage": current_stage,
            "conversation_stage": current_stage,
            "stage_transition_ready": (conversation_state or {}).get("stage_transition_ready", False),
        })
        debug_info = {
            **state.get("debug_info", {}),
            "stage": state.get("current_stage", current_stage),
            "current_stage": state.get("current_stage", current_stage),
            "phase": state.get("phase"),
            "voice_metadata": state.get("voice_metadata", ""),
            "completion_status": state.get("completion_status", False),
        }
        return state["response"], state["completion_status"], debug_info
