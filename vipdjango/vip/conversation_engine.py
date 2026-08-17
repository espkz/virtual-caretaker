import json
import logging
import re
from difflib import SequenceMatcher

from .conversation_graph import MAX_TURNS, TARGET_TURNS, build_conversation_graph
from .conversation_scenario import Scenario, parse_scenario_prompt


logger = logging.getLogger(__name__)


_EMBEDDED_VOICE_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*$", flags=re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9']+")

_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "could", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on",
    "or", "should", "the", "them", "there", "this", "to", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
}
_CLARIFICATION_MARKERS = (
    "what do you mean",
    "can you clarify",
    "i don't understand",
    "i do not understand",
    "could you explain that",
    "what does that mean",
)
_STAGE_ORDER = {"beginning": 0, "middle": 1, "ending": 2}


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


def _question_parts(text):
    """Return spoken questions, without treating the whole transcript as a topic."""
    return [
        part.strip()
        for part in re.split(r"(?<=\?)\s+", text or "")
        if "?" in part and part.strip(" ?")
    ]


def _question_tokens(text):
    return {
        token
        for token in _WORD_RE.findall((text or "").lower())
        if token not in _QUESTION_STOP_WORDS and len(token) > 2
    }


def _question_similarity(left, right):
    """Measure whether two questions are about the same concern.

    This intentionally uses a conservative lexical signal. The transcript is
    still supplied in full to the model; this helper only identifies a likely
    repeated question strongly enough to trigger a repair pass.
    """
    left_tokens = _question_tokens(left)
    right_tokens = _question_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    sequence = SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio()
    return max(overlap, sequence)


def _is_clarification_request(text):
    normalized = " ".join((text or "").lower().split())
    if any(marker in normalized for marker in _CLARIFICATION_MARKERS):
        return True
    return normalized.endswith("?") and len(_question_tokens(normalized)) <= 3


def _answered_questions(history):
    """List prior character questions that received a substantive learner reply."""
    answered = []
    for index, message in enumerate(history):
        if message.get("role") != "assistant":
            continue
        questions = _question_parts(message.get("content", ""))
        if not questions:
            continue
        next_user = next(
            (
                item.get("content", "")
                for item in history[index + 1:]
                if item.get("role") == "user"
            ),
            "",
        )
        if next_user and not _is_clarification_request(next_user):
            answered.extend(questions)
    return answered


def _conversation_memory(history):
    """Build a small, derived ledger so the model can use history as state."""
    answered = _answered_questions(history)
    if not answered:
        return "No earlier character question has been substantively answered yet."

    repeated_topics = []
    for index, question in enumerate(answered):
        if any(
            _question_similarity(question, earlier) >= 0.50
            for earlier in answered[:index]
        ):
            repeated_topics.append(question)

    lines = [
        "The learner has already responded to these character questions; do not ask the same concern again unless the learner explicitly says it was not understood:",
    ]
    lines.extend(f"- {question[:220]}" for question in answered[-12:])
    if repeated_topics:
        lines.append(
            "Some of those concerns have already been revisited. Acknowledge the latest answer, update your understanding, and move to a related unresolved concern or a natural stage transition."
        )
    return "\n".join(lines)


def _repeated_answered_question(dialogue, history):
    answered = _answered_questions(history)
    for question in _question_parts(dialogue):
        matches = [
            earlier for earlier in answered
            if _question_similarity(question, earlier) >= 0.50
        ]
        if matches:
            return question, matches[-1]
    return "", ""


def _remove_questions(dialogue):
    """Safe last-resort fallback if a repair response repeats an answered question."""
    pieces = re.split(r"(?<=[.!])\s+|(?<=\?)\s+", (dialogue or "").strip())
    kept = []
    for piece in pieces:
        if "?" in piece:
            break
        if piece.strip():
            kept.append(piece.strip())
    return " ".join(kept) or "Okay, I understand. I'll keep that in mind."


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
            "- Do not teach, coach, evaluate, or optimize the learner; if the character asks for information, ask as the character and wait for the learner's answer.\n"
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
            "CONVERSATION MEMORY:\n"
            f"{_conversation_memory(state.get('history', []))}\n"
            "Stage contract: scenario cues are semantic signals, not a checklist or a queue of required questions. "
            "Judge whether the conversation has served the purpose of the current stage. Beginning has no minimum "
            "length: once the learner has created a calm setting and addressed the character's immediate concern "
            "with appropriate honesty, move to Middle even if some example cues were not spoken verbatim. Do not "
            "delay a transition to cover every bullet, and do not begin technical training while important "
            "character concerns remain unresolved.\n"
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
            "Questions are optional. First acknowledge how the learner's answer changes your understanding; then "
            "move to a related concern or a new stage when appropriate. Do not ask a question merely to keep the "
            "conversation going, and do not repeat an answered question because the answer was imperfect. Respond "
            "as a human being in the character's situation, not as an assistant optimizing a communication plan. "
            "Return JSON fields dialogue, voice, stage, stage_transition_ready, complete, "
            "stop_requested."
        )

    def _request_llm_turn(self, state, repair_instruction=""):
        from openai import OpenAI

        scenario = state["scenario"]
        system_prompt = self._system_prompt(scenario, state)
        if repair_instruction:
            system_prompt = f"{system_prompt}\n\nREPAIR INSTRUCTION:\n{repair_instruction}"
        payload = [{"role": "system", "content": system_prompt}, *state["history"]]
        logger.debug("LLM context stage=%s turn=%s character=%s learner=%s payload=\n%s", state["current_stage"], state["current_turn"], scenario["character"], scenario["learner"], json.dumps(payload, ensure_ascii=False, indent=2))
        response = OpenAI(api_key=self.api_key).responses.create(
            model=self.model,
            text={"format": {"type": "json_schema", "name": "character_turn", "schema": {"type": "object", "properties": {"dialogue": {"type": "string"}, "voice": {"type": "string"}, "stage": {"type": "string", "enum": ["beginning", "middle", "ending"]}, "stage_transition_ready": {"type": "boolean"}, "complete": {"type": "boolean"}, "stop_requested": {"type": "boolean"}}, "required": ["dialogue", "voice", "stage", "stage_transition_ready", "complete", "stop_requested"], "additionalProperties": False}, "strict": True}},
            input=payload,
        )
        return json.loads(response.output_text or "{}")

    def _llm_turn(self, state):
        scenario = state["scenario"]
        result = self._request_llm_turn(state)
        interrupted = bool(result.get("stop_requested"))
        if interrupted:
            return "", "", state["current_stage"], True, True, {"reason": "interrupt"}
        parsed_scenario = Scenario(**scenario)
        proposed_stage = result.get("stage", state["current_stage"])
        transition_ready = bool(result.get("stage_transition_ready"))
        current_stage = state["current_stage"]
        if proposed_stage not in _STAGE_ORDER:
            proposed_stage = current_stage
            transition_ready = False
        elif _STAGE_ORDER[proposed_stage] < _STAGE_ORDER[current_stage]:
            # Stage is monotonic even if the model accidentally reports a
            # previous stage after seeing a long transcript.
            proposed_stage = state["current_stage"]
            transition_ready = False
        elif proposed_stage != current_stage:
            # The selected forward stage is itself the semantic decision. The
            # boolean is retained as diagnostic output for compatibility, but
            # a false value must not discard a valid stage selection.
            transition_ready = True
        model_requested_completion = bool(result.get("complete"))
        # A completion flag is meaningful only with an explicit semantic
        # transition to Ending. This prevents the soft target from becoming
        # an implicit closing command.
        complete = model_requested_completion and proposed_stage == "ending" and transition_ready
        raw_dialogue = (result.get("dialogue", "") or "").strip()

        repeated_question, answered_question = _repeated_answered_question(raw_dialogue, state["history"])
        repetition_repaired = False
        if repeated_question:
            repair_instruction = (
                "The draft repeats an answered character concern. Do not ask that question again. "
                f"The repeated draft question was: {repeated_question} "
                f"An earlier answered question was: {answered_question} "
                "Acknowledge the learner's latest answer, let it change Rachel's understanding, and move to a "
                "related unresolved concern or a natural stage transition. Questions are optional. Return the same "
                "JSON schema."
            )
            repaired = self._request_llm_turn(state, repair_instruction)
            if repaired.get("stop_requested"):
                # A repair response is not allowed to turn a normal learner
                # turn into an interrupt. Keep the character response and
                # remove the repeated question locally.
                raw_dialogue = _remove_questions(raw_dialogue)
            else:
                result = repaired
                repetition_repaired = True
                proposed_stage = result.get("stage", current_stage)
                transition_ready = bool(result.get("stage_transition_ready"))
                if proposed_stage not in _STAGE_ORDER or _STAGE_ORDER[proposed_stage] < _STAGE_ORDER[current_stage]:
                    proposed_stage = current_stage
                    transition_ready = False
                elif proposed_stage != current_stage:
                    transition_ready = True
                model_requested_completion = bool(result.get("complete"))
                complete = model_requested_completion and proposed_stage == "ending" and transition_ready
                raw_dialogue = (result.get("dialogue", "") or "").strip()
                if _repeated_answered_question(raw_dialogue, state["history"])[0]:
                    raw_dialogue = _remove_questions(raw_dialogue)

        dialogue, embedded_voice = split_dialogue_and_voice(raw_dialogue)
        voice_value = result.get("voice", "") or embedded_voice
        if embedded_voice and (not voice_value or voice_value.strip("[] ").lower() in {"male", "female", "voice"}):
            # Preserve a useful style if a malformed response supplies only
            # the gender in the structured field but puts the style in the
            # legacy bracket position.
            voice_value = embedded_voice

        if complete and parsed_scenario.closing:
            dialogue = parsed_scenario.closing

        # The safety cap is an operational boundary, not a conversational
        # ending. Never substitute the scenario Closing merely because a
        # counter was reached; a closing is emitted only after the model has
        # semantically moved to Ending.
        at_safety_cap = state.get("current_turn", 0) >= state.get("max_turns", MAX_TURNS)
        reason = "llm_turn_at_safety_cap" if at_safety_cap else "llm_turn"
        return dialogue, _voice_metadata(voice_value, parsed_scenario), proposed_stage, complete, False, {
            "reason": reason,
            "stage_transition_ready": transition_ready,
            "completion_requested": model_requested_completion,
            "completion_rejected": model_requested_completion and not complete,
            "embedded_voice_removed": bool(embedded_voice),
            "repetition_repaired": repetition_repaired,
            "repeated_question_blocked": bool(repeated_question),
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
