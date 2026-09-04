import json
import logging
import math
import re
import threading
from difflib import SequenceMatcher

from .conversation_graph import MAX_TURNS, TARGET_TURNS, build_conversation_graph
from .conversation_scenario import Scenario, parse_scenario_prompt


logger = logging.getLogger(__name__)

DEFAULT_INTRODUCTION = "Please type to continue speaking with the Assistant."


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
_ENDING_SIGNAL_PHRASES = (
    "ready",
    "prepared",
    "start learning",
    "begin learning",
    "get started",
    "start now",
    "show me how",
    "i'd like to learn",
    "i would like to learn",
    "no more questions",
    "that's all",
    "thank you for",
    "thanks for",
    "i appreciate",
    "nice speaking",
    "good day",
    "enjoyed talking",
    "enjoyed our conversation",
    "done for now",
    "take care",
    "goodbye",
    "see you",
)
_ENDING_NEGATIVE_PHRASES = (
    "not ready",
    "not yet",
    "still need to",
    "more questions",
    "one more question",
    "can't learn this",
    "cannot learn this",
)
_HISTORY_LABEL_RE = re.compile(r"^\s*(?:LEARNER|SIMULATED CHARACTER):\s*", flags=re.IGNORECASE)


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


def format_history_content(role, content):
    """Add a simulation-speaker label without changing the native API role."""
    value = (content or "").strip()
    if _HISTORY_LABEL_RE.match(value):
        return value
    label = "LEARNER" if role == "user" else "SIMULATED CHARACTER"
    return f"{label}:\n{value}"


def _without_history_label(text):
    return _HISTORY_LABEL_RE.sub("", (text or "").strip(), count=1)


def _history(messages, scenario=None):
    """Convert spoken transcript to native LLM roles with clear speaker labels.

    The application-owned Introduction is already represented by the active
    scenario and is not dialogue. Omitting it avoids sending the same setup
    instructions on every later turn while retaining all spoken history.
    """
    introduction = ""
    if scenario is not None:
        introduction = (
            scenario.get("introduction", "")
            if isinstance(scenario, dict)
            else getattr(scenario, "introduction", "")
        ).strip()
    introduction_skipped = False
    result = []
    for message in messages:
        sender = str(message.sender)
        if sender not in {"student", "assistant"}:
            continue
        content = (message.content or "").strip()
        if sender == "assistant":
            content, _ = split_dialogue_and_voice(content)
            if introduction and not introduction_skipped and content == introduction:
                introduction_skipped = True
                continue
        role = "user" if sender == "student" else "assistant"
        result.append({"role": role, "content": format_history_content(role, content)})
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
    text = _without_history_label(text)
    return [
        part.strip()
        for part in re.split(r"(?<=\?)\s+", text or "")
        if "?" in part and part.strip(" ?")
    ]


def _question_tokens(text):
    return {
        token
        for token in _WORD_RE.findall(_without_history_label(text).lower())
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


def _scenario_objectives(scenario):
    """Return valid scenario objective records without trusting model output."""
    records = []
    for objective in (scenario or {}).get("objectives", []) or []:
        if not isinstance(objective, dict):
            continue
        objective_id = str(objective.get("id", "")).strip()
        if objective_id:
            records.append(
                {
                    "id": objective_id,
                    "description": str(objective.get("description", "")).strip(),
                    "possible_expressions": str(objective.get("possible_expressions", "")).strip(),
                    "resolved_when": str(objective.get("resolved_when", "")).strip(),
                }
            )
    return records


def _valid_objective_values(values, valid_ids):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    return [value for value in values if isinstance(value, str) and value in valid_ids]


def _initial_objective_progress(scenario, state):
    objective_ids = [objective["id"] for objective in _scenario_objectives(scenario)]
    valid_ids = set(objective_ids)
    covered = list(dict.fromkeys(_valid_objective_values(state.get("covered_objectives"), valid_ids)))
    unresolved = list(dict.fromkeys(_valid_objective_values(state.get("unresolved_objectives"), valid_ids)))
    if objective_ids and not covered and not unresolved:
        unresolved = list(objective_ids)
    unresolved = [value for value in unresolved if value not in covered]
    active = state.get("active_objective", "")
    if active not in unresolved:
        active = unresolved[0] if unresolved else ""
    return {
        "active_objective": active,
        "covered_objectives": covered,
        "unresolved_objectives": unresolved,
        "ending_ready": bool(state.get("ending_ready", False)),
    }


def _recent_topics(history, previous=None):
    """Keep a small, generic topic trail without replacing the full transcript."""
    topics = []
    seen = set()
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue
        content = _without_history_label(message.get("content", ""))
        values = _question_parts(content) or [content]
        for value in values:
            value = " ".join(value.split()).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            topics.append(value[:220])
            if len(topics) >= 6:
                return list(reversed(topics))
    if topics:
        return list(reversed(topics))
    return list(previous or [])[-6:]


def _objective_progress_context(scenario, state):
    objectives = _scenario_objectives(scenario)
    if not objectives:
        return "No scenario-defined conversation objectives are configured. Follow the scenario behavior and do not invent educational objectives."
    progress = _initial_objective_progress(scenario, state)

    lines = [
        "Objectives describe underlying concerns, not a rigid checklist or required question sequence.",
        "Objective records are internal progress labels. Possible expressions are examples of concerns the Simulated Character may express when the Learner's latest response leaves them unresolved. They are not questions for the Learner's role, not a facilitator script, and not required wording. Judge the learner's meaning. One learner response may cover multiple objectives, and an objective may be covered without its example question being spoken.",
    ]
    for objective in objectives:
        lines.append(f"- {objective['id']}: {objective['description']}")
        if objective["possible_expressions"]:
            lines.append(f"  Possible expressions: {objective['possible_expressions']}")
        if objective["resolved_when"]:
            lines.append(f"  Resolved when: {objective['resolved_when']}")
    lines.extend(
        [
            f"Active objective: {progress['active_objective'] or 'none selected'}",
            f"Covered objectives: {', '.join(progress['covered_objectives']) or 'none'}",
            f"Unresolved objectives: {', '.join(progress['unresolved_objectives']) or 'none'}",
            f"Recent learner topics/questions: {' | '.join(state.get('recent_topics', [])) or 'none recorded'}",
            "Treat the progress fields as lightweight guidance. Acknowledge the learner naturally, do not force every objective, and do not ask every possible question.",
        ]
    )
    return "\n".join(lines)


def _normalize_objective_progress(scenario, state, result, history):
    """Validate one model progress snapshot and keep progress monotonic."""
    previous = _initial_objective_progress(scenario, state)
    valid_ids = {objective["id"] for objective in _scenario_objectives(scenario)}
    covered = list(previous["covered_objectives"])
    if "covered_objectives" in result:
        covered.extend(_valid_objective_values(result.get("covered_objectives"), valid_ids))
    covered = list(dict.fromkeys(covered))

    unresolved = list(previous["unresolved_objectives"])
    if "unresolved_objectives" in result:
        unresolved.extend(_valid_objective_values(result.get("unresolved_objectives"), valid_ids))
    unresolved = [value for value in dict.fromkeys(unresolved) if value not in covered]

    active = result.get("active_objective", previous["active_objective"])
    if active not in unresolved:
        active = unresolved[0] if unresolved else ""
    ending_ready = (
        bool(result["ending_ready"])
        if "ending_ready" in result
        else previous["ending_ready"]
    )
    return {
        "active_objective": active,
        "covered_objectives": covered,
        "unresolved_objectives": unresolved,
        "recent_topics": _recent_topics(history, state.get("recent_topics")),
        "ending_ready": ending_ready,
    }


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


def _looks_like_ending_candidate(dialogue, latest_learner=""):
    """Cheap trigger for the semantic ending verifier, not the decision itself."""
    text = " ".join((dialogue or "", latest_learner or "")).lower()
    if any(marker in text for marker in _ENDING_NEGATIVE_PHRASES):
        return False
    return any(marker in text for marker in _ENDING_SIGNAL_PHRASES)


def _context_measurement(payload, request_kind):
    """Measure the exact input payload shape without retaining its contents."""
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    content_chars = sum(
        len(message.get("content", ""))
        for message in payload
        if isinstance(message, dict) and isinstance(message.get("content", ""), str)
    )
    content_bytes = sum(
        len(message.get("content", "").encode("utf-8"))
        for message in payload
        if isinstance(message, dict) and isinstance(message.get("content", ""), str)
    )
    serialized_bytes = len(serialized.encode("utf-8"))
    return {
        "request_kind": request_kind,
        "message_count": len(payload),
        "content_chars": content_chars,
        "content_bytes": content_bytes,
        "serialized_chars": len(serialized),
        "serialized_bytes": serialized_bytes,
        # This is deliberately labeled as an estimate. Provider tokenization
        # is reported below when the API returns usage data.
        "estimated_tokens": math.ceil(serialized_bytes / 4),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def _usage_value(usage, name):
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    return value if isinstance(value, int) else None


def _add_context_measurement(context_measurements, payload, request_kind):
    if not isinstance(context_measurements, list):
        return None
    measurement = _context_measurement(payload, request_kind)
    context_measurements.append(measurement)
    return measurement


def _attach_response_usage(measurement, response):
    if not measurement:
        return
    usage = getattr(response, "usage", None)
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = _usage_value(usage, name)
        if value is not None:
            measurement[name] = value


def _log_context_measurement(measurement):
    if measurement:
        logger.info("llm_context %s", json.dumps(measurement, ensure_ascii=False, sort_keys=True))


class ConversationEngine:
    def __init__(self, global_prompt, api_key, model="gpt-5-nano"):
        self.global_prompt = global_prompt.strip()
        self.api_key = api_key
        self.model = model
        self._client = None
        self._client_lock = threading.Lock()
        self.graph = build_conversation_graph(self._llm_turn)

    def _openai_client(self):
        """Reuse one HTTP client across requests made during this turn."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI

                    self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _interrupt(self, message, timing_callback=None, context_measurements=None):
        payload = [
            {"role": "system", "content": "Return true only when the learner clearly asks to stop or end the conversation now. Return false for ordinary roleplay, thanks with a question, or discussion of an ending."},
            {"role": "user", "content": message},
        ]
        measurement = _add_context_measurement(context_measurements, payload, "interrupt")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Interrupt payload: %s",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        if timing_callback:
            timing_callback("interrupt_request_start")
        response = self._openai_client().responses.create(
            model=self.model,
            text={"format": {"type": "json_schema", "name": "interrupt", "schema": {"type": "object", "properties": {"stop_requested": {"type": "boolean"}}, "required": ["stop_requested"], "additionalProperties": False}, "strict": True}},
            input=payload,
        )
        _attach_response_usage(measurement, response)
        _log_context_measurement(measurement)
        if timing_callback:
            timing_callback("interrupt_request_complete")
        return bool(json.loads(response.output_text or "{}").get("stop_requested"))

    def _system_prompt(self, scenario, state):
        stage = state.get("current_stage", "beginning")
        behavior = scenario[stage]
        cues = {
            "beginning": scenario["beginning_to_middle_cues"],
            "middle": scenario["middle_to_ending_cues"],
            "ending": scenario["end_of_conversation_cues"],
        }[stage]
        background_context = (scenario.get("background_context") or "").strip()
        scenario_context = (
            "SCENARIO REFERENCE DATA (parsed from the role prompt; not dialogue or a second set of speaker instructions):\n"
            f"Background and context: {background_context or 'No additional background context is provided.'}\n"
            "Scenario reference data and author notes are internal simulation data, not automatic character knowledge or dialogue. Use facts only when the Simulated Character would reasonably know them and they are relevant to the Learner's latest turn or the character's own concern. Do not recite scenario facts or introduce clinical information unprompted.\n\n"
        )
        return (
            f"{self.global_prompt}\n\n"
            f"{scenario_context}"
            "PARTICIPANT OWNERSHIP (immutable):\n"
            f"SIMULATED CHARACTER / ASSISTANT CHARACTER (## Role): {scenario['character']}\n"
            f"LEARNER / USER (## Learner Role): {scenario['learner']}\n"
            "You are the Simulated Character and generate dialogue only for that character. The human-controlled participant is the Learner. The Learner drives the encounter; respond to what the Learner actually says. Do not turn scenario topics into a questionnaire for the Learner or ask the Learner to teach, explain, or perform clinical reasoning. If a question is natural, it must be a brief concern from the Simulated Character's perspective. Never speak, think, act, teach, advise, or answer on behalf of the Learner, nurse, instructor, facilitator, or any other role. Never invent learner actions, thoughts, feelings, dialogue, or identity, and never take on a different role.\n\n"
            "HISTORY SPEAKER LABELS: prior transcript entries retain native API roles (`assistant` for the Simulated Character and `user` for the Learner). Their content also begins with `SIMULATED CHARACTER:` or `LEARNER:`; these are simulation labels, not additional speakers.\n\n"
            f"CURRENT STAGE: {stage}\n"
            "ACTIVE-STAGE CHARACTER GUIDANCE (respond as the Simulated Character; not a learner questionnaire or facilitator task list):\n"
            f"{behavior}\n\n"
            "STAGE / TRANSITION STATE (progression guidance only; it never changes speaker ownership):\n"
            f"{cues}\n"
            f"SCENARIO META GUIDANCE (character-side reference only):\n{scenario['meta']}\n"
            f"OBJECTIVE PROGRESS (lightweight guidance):\n{_objective_progress_context(scenario, state)}\n"
            "PHASE CONTRACT: Beginning establishes rapport and clarifies the character's immediate concern; it has no minimum length and may transition early when that purpose is served. Middle uses unresolved objectives and the learner's latest meaning to choose the next relevant concern; do not remain on a covered theme merely because the phase is Middle. Ending is appropriate when the scenario's endpoint is naturally ready, the learner signals completion, and/or the character's important concerns have been reasonably addressed or can be acknowledged naturally. Do not force every objective or use turn count as the reason to end.\n"
            f"Voice: {scenario['voice_gender']} voice; baseline style: {scenario['voice_style']}\n"
            "CONVERSATION MEMORY:\n"
            f"{_conversation_memory(state.get('history', []))}\n"
            "Stage contract: scenario cues are semantic signals, not a checklist or a queue of required questions. "
            "Judge whether the conversation has served the purpose of the current stage. Beginning has no minimum "
            "length: once the learner has created a calm setting and addressed the character's immediate concern "
            "with appropriate honesty, move to Middle even if some example cues were not spoken verbatim. Do not "
            "delay a transition to cover every bullet, and do not begin technical training while important "
            "character concerns remain unresolved.\n"
            "ENDING CONTRACT: completion is semantic. The canonical Closing is an example/output template, not an "
            "exact sentence that the character must reproduce. If the character's response meaningfully fulfills the "
            "scenario's ending purpose in different words, select Ending and complete the conversation. Do not "
            "complete for an ordinary acknowledgement, a polite thanks that leaves a substantive question open, or "
            "a response that merely continues the interaction.\n"
            "TURN-BUDGET CONTRACT: "
            f"Current learner turn: {state.get('current_turn', 0)}; soft target: {state.get('target_turns', TARGET_TURNS)}; "
            f"turns remaining before the safety cap: {state.get('turns_remaining', MAX_TURNS)}; phase: {state.get('phase', 'normal')}. "
            "The target is an advisory pressure against indefinite looping, not a minimum, schedule, stage rule, or "
            "termination trigger. Never complete or emit the Closing merely because a number was reached. Do not "
            "add filler questions or repeat answered topics.\n"
            "OUTPUT CONTRACT: Return JSON only. `dialogue` contains only the character's spoken words; never put "
            "voice, emotion, stage notes, brackets, narration, or metadata there. `voice` is concise TTS style "
            "metadata without brackets or the word 'voice'. Questions are optional. Acknowledge the learner's answer, "
            "then move to a related concern or a new stage when appropriate. Return fields dialogue, voice, stage, "
            "stage_transition_ready, complete, stop_requested, active_objective, covered_objectives, "
            "unresolved_objectives, ending_ready. For objective fields, return the complete current progress "
            "snapshot using only the scenario objective IDs; preserve genuinely unresolved concerns, allow one "
            "answer to cover multiple objectives or a future objective early, and do not force every objective "
            "or example question. Set ending_ready only when a natural endpoint is available; an unresolved "
            "objective may remain when the character has acknowledged it appropriately, but do not complete "
            "while an important unresolved concern still needs attention."
        )

    def _request_llm_turn(self, state, repair_instruction=""):
        scenario = state["scenario"]
        system_prompt = self._system_prompt(scenario, state)
        if repair_instruction:
            system_prompt = f"{system_prompt}\n\nREPAIR INSTRUCTION:\n{repair_instruction}"
        history = [
            {
                **message,
                "content": format_history_content(message.get("role"), message.get("content", "")),
            }
            for message in state["history"]
        ]
        payload = [{"role": "system", "content": system_prompt}, *history]
        request_kind = "repair" if repair_instruction else "generation"
        measurement = _add_context_measurement(
            state.get("context_measurements"),
            payload,
            request_kind,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM context stage=%s turn=%s character=%s learner=%s payload=\n%s",
                state["current_stage"],
                state["current_turn"],
                scenario["character"],
                scenario["learner"],
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        timing_callback = state.get("timing_callback")
        if timing_callback:
            timing_callback("gpt_request_start")

        request_kwargs = {
            "model": self.model,
            "text": {"format": {"type": "json_schema", "name": "character_turn", "schema": {"type": "object", "properties": {"dialogue": {"type": "string"}, "voice": {"type": "string"}, "stage": {"type": "string", "enum": ["beginning", "middle", "ending"]}, "stage_transition_ready": {"type": "boolean"}, "complete": {"type": "boolean"}, "stop_requested": {"type": "boolean"}, "active_objective": {"type": "string"}, "covered_objectives": {"type": "array", "items": {"type": "string"}}, "unresolved_objectives": {"type": "array", "items": {"type": "string"}}, "ending_ready": {"type": "boolean"}}, "required": ["dialogue", "voice", "stage", "stage_transition_ready", "complete", "stop_requested", "active_objective", "covered_objectives", "unresolved_objectives", "ending_ready"], "additionalProperties": False}, "strict": True}},
            "input": payload,
        }

        response = self._openai_client().responses.create(**request_kwargs)
        _attach_response_usage(measurement, response)
        _log_context_measurement(measurement)
        if timing_callback and response.output_text:
            timing_callback("gpt_first_output")
        if timing_callback:
            timing_callback("gpt_completion")
        return json.loads(response.output_text or "{}")

    def _verify_semantic_ending(self, state, dialogue, proposed_stage):
        """Conservatively verify that a candidate response fulfills the endpoint."""
        scenario = state["scenario"]
        payload = [
            {
                "role": "system",
                "content": (
                    "You are a conservative semantic endpoint verifier for a roleplay conversation. "
                    "Do not write character dialogue. Decide whether the candidate assistant response fulfills "
                    "the scenario's intended ending now. Do not compare strings or require the canonical Closing "
                    "word-for-word. Treat the Ending and transition cues as guidance about purpose, not a literal "
                    "script. Return true only when the character has clearly reached an endpoint such as being "
                    "ready to begin the next activity, expressing final gratitude/farewell, or satisfying another "
                    "explicit endpoint in the Ending guidance. A generic acknowledgement, ordinary thanks, an "
                    "unresolved substantive concern, or an ongoing request for information is false unless the "
                    "scenario explicitly defines that request as the endpoint. If the Ending guidance says to stop "
                    "before simulating the next activity, readiness to begin that activity is an endpoint even if "
                    "the candidate mentions what would happen next. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_stage": proposed_stage,
                        "ending_guidance": scenario["ending"],
                        "transition_cues": scenario["middle_to_ending_cues"],
                        "canonical_closing_example": scenario["closing"],
                        "objective_progress": {
                            "active_objective": state.get("active_objective", ""),
                            "covered_objectives": state.get("covered_objectives", []),
                            "unresolved_objectives": state.get("unresolved_objectives", []),
                            "ending_ready": state.get("ending_ready", False),
                        },
                        "conversation_history": state.get("history", []),
                        "candidate_character_response": dialogue,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        measurement = _add_context_measurement(
            state.get("context_measurements"),
            payload,
            "semantic_ending_verifier",
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Ending verification payload:\n%s",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
        try:
            response = self._openai_client().responses.create(
                model=self.model,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "semantic_ending_check",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ending_satisfied": {"type": "boolean"},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                "reason": {"type": "string"},
                            },
                            "required": ["ending_satisfied", "confidence", "reason"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                },
                input=payload,
            )
            _attach_response_usage(measurement, response)
            result = json.loads(response.output_text or "{}")
            check = {
                "attempted": True,
                "satisfied": bool(result.get("ending_satisfied")),
                "confidence": result.get("confidence", "low"),
                "reason": (result.get("reason") or "no verifier reason")[:300],
            }
        except Exception as exc:
            logger.exception("Semantic ending verification failed")
            check = {
                "attempted": True,
                "satisfied": False,
                "confidence": "low",
                "reason": f"verifier_error: {type(exc).__name__}",
            }
        _log_context_measurement(measurement)
        logger.debug(
            "Ending decision stage=%s candidate=%r satisfied=%s confidence=%s reason=%s",
            proposed_stage,
            dialogue,
            check["satisfied"],
            check["confidence"],
            check["reason"],
        )
        return check

    def _llm_turn(self, state):
        state.setdefault("context_measurements", [])
        scenario = state["scenario"]
        state.update(_initial_objective_progress(scenario, state))
        state["recent_topics"] = _recent_topics(state.get("history", []), state.get("recent_topics"))
        result = self._request_llm_turn(state)
        interrupted = bool(result.get("stop_requested"))
        if interrupted:
            return "", "", state["current_stage"], True, True, {
                "reason": "interrupt",
                **_normalize_objective_progress(scenario, state, result, state.get("history", [])),
                "llm_context": list(state.get("context_measurements", [])),
            }
        parsed_scenario = Scenario.from_state(scenario)
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
                "Acknowledge the learner's latest answer, let it change the simulated character's understanding, and move to a "
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

        objective_progress = _normalize_objective_progress(
            scenario,
            state,
            result,
            state.get("history", []),
        )
        state.update(objective_progress)
        objective_ready = not objective_progress["unresolved_objectives"] or objective_progress["ending_ready"]
        complete = complete and objective_ready

        ending_check = {
            "attempted": False,
            "satisfied": False,
            "confidence": "none",
            "reason": "no ending candidate signal",
        }
        latest_learner = ""
        if state.get("history"):
            latest_learner = next(
                (
                    item.get("content", "")
                    for item in reversed(state["history"])
                    if item.get("role") == "user"
                ),
                "",
            )
        if (
            not complete
            and proposed_stage in {"middle", "ending"}
            and (
                _looks_like_ending_candidate(raw_dialogue, latest_learner)
                or state.get("phase") in {"closure_preference", "closure_flexibility"}
            )
        ):
            ending_check = self._verify_semantic_ending(state, raw_dialogue, proposed_stage)
            if ending_check["satisfied"]:
                if objective_ready:
                    proposed_stage = "ending"
                    transition_ready = True
                    complete = True
                else:
                    ending_check["satisfied"] = False
                    ending_check["reason"] = "objective_progress_not_ready"

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
        if ending_check["satisfied"]:
            reason = "semantic_ending_verified"
        else:
            reason = "llm_turn_at_safety_cap" if at_safety_cap else "llm_turn"
        return dialogue, _voice_metadata(voice_value, parsed_scenario), proposed_stage, complete, False, {
            "reason": reason,
            "stage_transition_ready": transition_ready,
            "completion_requested": model_requested_completion,
            "completion_rejected": model_requested_completion and not complete,
            "embedded_voice_removed": bool(embedded_voice),
            "repetition_repaired": repetition_repaired,
            "repeated_question_blocked": bool(repeated_question),
            "ending_check": ending_check,
            **objective_progress,
            "llm_context": list(state.get("context_measurements", [])),
        }

    def respond(self, role_text, messages, conversation_state=None, timing_callback=None):
        scenario = parse_scenario_prompt(role_text)
        learner_messages = [m for m in messages if m.sender == "student"]
        turn = len(learner_messages)
        if turn == 0:
            return (scenario.introduction or DEFAULT_INTRODUCTION), False, {
                "stage": "beginning",
                "current_stage": "beginning",
                "voice_metadata": "",
                "reason": "introduction",
            }
        latest = learner_messages[-1].content
        if turn == 1 and scenario.opening_line:
            opening_already_emitted = any(
                message.sender == "assistant"
                and split_dialogue_and_voice(message.content)[0].strip() == scenario.opening_line.strip()
                for message in messages
            )
            if not opening_already_emitted:
                try:
                    context_measurements = []
                    if self._interrupt(
                        latest,
                        timing_callback=timing_callback,
                        context_measurements=context_measurements,
                    ):
                        return "", True, {
                            "stage": "ending",
                            "current_stage": "ending",
                            "interrupt_requested": True,
                            "reason": "interrupt",
                            "llm_context": context_measurements,
                        }
                except Exception:
                    logger.exception("Interrupt classification failed")
                return scenario.opening_line, False, {
                    "stage": "beginning",
                    "current_stage": "beginning",
                    "voice_metadata": scenario.voice_metadata(),
                    "reason": "fixed_opening_line",
                    "llm_context": context_measurements,
                }
        current_stage = (conversation_state or {}).get("current_stage")
        if current_stage not in {"beginning", "middle", "ending"}:
            current_stage = "beginning"
        scenario_state = scenario.to_state()
        objective_progress = _initial_objective_progress(scenario_state, conversation_state or {})
        graph_input = {
            "scenario": scenario_state,
            "history": _history(messages, scenario),
            **objective_progress,
            "recent_topics": _recent_topics(_history(messages, scenario)),
            "current_turn": turn,
            "target_turns": TARGET_TURNS,
            "max_turns": MAX_TURNS,
            "current_stage": current_stage,
            "conversation_stage": current_stage,
            "stage_transition_ready": (conversation_state or {}).get("stage_transition_ready", False),
            "timing_callback": timing_callback,
            "context_measurements": [],
        }

        # Normal turns already request semantic stop intent in the structured
        # response. A second classifier request only added latency and forced
        state = self.graph.invoke(graph_input)
        debug_info = {
            **state.get("debug_info", {}),
            "stage": state.get("current_stage", current_stage),
            "current_stage": state.get("current_stage", current_stage),
            "phase": state.get("phase"),
            "voice_metadata": state.get("voice_metadata", ""),
            "completion_status": state.get("completion_status", False),
            "active_objective": state.get("active_objective", ""),
            "covered_objectives": list(state.get("covered_objectives", [])),
            "unresolved_objectives": list(state.get("unresolved_objectives", [])),
            "recent_topics": list(state.get("recent_topics", [])),
            "ending_ready": state.get("ending_ready", False),
            "llm_context": list(state.get("context_measurements", [])),
        }
        return state["response"], state["completion_status"], debug_info
