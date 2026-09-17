"""Scenario-grounded dialogue with application-owned pacing and termination.

Question IDs track concerns, not a script. Assessment and speech share one model
request; only the application can commit progress or finish an encounter.
"""

from copy import deepcopy
import logging
import random
import re

from .conversation_graph import MAX_TURNS


logger = logging.getLogger(__name__)

PAUSE_CLOSING = "I need to pause here and take some time to process this."
SUPPORT_CLOSING = "I'm not ready to go ahead yet. I need some time to take in what we've discussed."
_LEADING_AUDIO_TAGS_RE = re.compile(
    r"^(?P<tags>(?:\[[a-z][a-z'-]*(?:[ \t]+[a-z][a-z'-]*){0,3}\][ \t]*)+)"
)


def enabled(scenario):
    return (
        scenario.simulation_mode == "roleplay"
        and bool(scenario.topics)
        and all(topic.possible_expressions for topic in scenario.topics)
    )


def question_bank(scenario):
    return [
        {"id": f"{i}:{j}", "theme": i, "text": expression}
        for i, topic in enumerate(scenario.topics)
        for j, expression in enumerate(topic.possible_expressions)
    ]


def initial_state(scenario, messages):
    """Recover identifiable concerns from older transcripts conservatively."""
    state = {"asked": [], "pending": None, "clarified": [], "unresolved": []}
    for message in messages:
        if message.sender != "assistant":
            continue
        text = message.content.strip()
        if scenario.opening_line and text == scenario.opening_line.strip():
            question = {"id": "opening", "theme": 0, "text": scenario.opening_line}
        else:
            question = next((q for q in question_bank(scenario) if q["text"] in text), None)
        if question and question["id"] not in [item["id"] for item in state["asked"]]:
            if state["pending"]:
                state["unresolved"].append(state["pending"]["id"])
            state["asked"].append(question)
            state["pending"] = question
    return state


def opening_state(scenario):
    question = {"id": "opening", "theme": 0, "text": scenario.opening_line}
    return {"asked": [question], "pending": question, "clarified": [], "unresolved": []}


def candidates(scenario, progress, include_next=False):
    used = {item["id"] for item in progress["asked"]} | set(progress.get("addressed", []))
    if "opening" in used:
        used.add("0:0")
    theme = progress.get("active_theme", 0)
    themes = {theme}
    if include_next:
        themes.update(range(theme))
        if sum(q["theme"] == theme for q in progress["asked"]) >= 2:
            themes.add(theme + 1)
    remaining = [
        q for q in question_bank(scenario)
        if q["theme"] in themes and q["id"] not in used
    ]
    order = {qid: index for index, qid in enumerate(progress.get("question_order", []))}
    remaining.sort(key=lambda q: order.get(q["id"], len(order)))

    options = {}
    for question in remaining:
        options.setdefault(question["theme"], question)
    counts = {
        i: sum(q["theme"] == i for q in progress["asked"])
        for i in options
    }
    return sorted(options.values(), key=lambda q: counts[q["theme"]])


def prepare(scenario, messages, persisted):
    progress = deepcopy(persisted or initial_state(scenario, messages))
    bank_ids = [q["id"] for q in question_bank(scenario)]
    if "question_order" not in progress:
        progress["question_order"] = list(bank_ids)
        random.SystemRandom().shuffle(progress["question_order"])
    else:
        progress["question_order"] = list(dict.fromkeys(
            qid for qid in progress["question_order"] if qid in bank_ids
        ))
        progress["question_order"].extend(
            qid for qid in bank_ids if qid not in progress["question_order"]
        )
    progress.setdefault("addressed", [])
    progress.setdefault("follow_ups", {})
    progress.setdefault("active_theme", (progress.get("pending") or {}).get("theme", 0))
    progress["turn"] = sum(m.sender == "student" for m in messages)

    window = max(1, (MAX_TURNS - 2) // len(scenario.topics))
    scheduled = min(len(scenario.topics) - 1, max(0, (progress["turn"] - 1) // window))
    theme = max(progress["active_theme"], scheduled)
    progress["active_theme"] = theme
    while theme < len(scenario.topics) - 1 and not candidates(scenario, progress):
        pending = progress.get("pending")
        if pending and pending["id"] not in progress["addressed"]:
            break
        theme += 1
        progress["active_theme"] = theme
    pending = progress.get("pending")
    progress["must_advance"] = bool(
        pending and progress["follow_ups"].get(pending["id"], 0) >= 1
    )
    return progress


def validate_dialogue(dialogue, messages):
    """Reject obvious role/dose leakage; semantic fidelity still needs review."""
    if not isinstance(dialogue, str) or not dialogue.strip() or len(dialogue) > 1600:
        raise ValueError("Invalid character dialogue")
    normalized = " ".join(dialogue.lower().replace("’", "'").split())
    forbidden = (
        "i'm not sure i understand. could you explain that concern in simpler terms",
        "i'd like to speak with someone from the care team",
        "i need to talk with someone else",
        "i need to speak with the hospice supervisor",
        "i am your nurse",
        "i'm your nurse",
        "as your nurse",
        "as a nurse",
        "you should give her",
        "you can give her",
        "i recommend administering",
    )
    if any(value in normalized for value in forbidden) or re.search(
        r"\b\d+(?:\.\d+)?\s*(?:mg|ml|milligrams?|milliliters?)\b", normalized
    ):
        raise ValueError("Character dialogue violated role boundaries")
    previous = next((m.content.strip() for m in reversed(messages) if m.sender == "assistant"), "")
    if dialogue.strip() == previous:
        raise ValueError("Character dialogue repeated the previous turn")
    return dialogue.strip()


def render_question(dialogue, question, status, turn):
    """The chosen faculty question and persisted ID must describe the same concern.

    Keep the generated answer when the nurse asks a question. For an ordinary
    explanation, retain only a short acknowledgement, then ask the fresh concern.
    Never append a second paraphrase of the previous question.
    """
    if dialogue.strip() == question["text"].strip():
        return question["text"]
    audio_tags = _LEADING_AUDIO_TAGS_RE.match(dialogue.strip())
    delivery = audio_tags.group("tags").strip() if audio_tags else ""
    statements = [s.strip() for s in re.split(r"(?<=[.!?])\s+", dialogue) if "?" not in s]
    if status == "learner_question" or turn == 1:
        prefix = " ".join(statements)
    elif status in {"addressed", "partial"}:
        prefix = delivery
    else:
        prefix = statements[0] if statements and len(statements[0].split()) <= 12 else ""
    return " ".join(value for value in (prefix, question["text"]) if value)


def respond(scenario, messages, persisted, assess):
    progress = prepare(scenario, messages, persisted)
    available = candidates(scenario, progress, include_next=True)
    pending = progress["pending"]
    for attempt in range(2):
        result = assess(scenario, messages, pending, available, progress)
        status = result.get("answer_status")
        if status not in {"addressed", "partial", "unclear", "unsafe", "learner_question"}:
            raise ValueError("Invalid core-question assessment")
        try:
            question_id = result.get("question_id", "")
            raw_dialogue = result.get("dialogue")
            selected = next((q for q in available if q["id"] == question_id), None)
            if raw_dialogue == "" and selected:
                raw_dialogue = selected["text"]
            dialogue = validate_dialogue(raw_dialogue, messages)
            repeats_pending = pending and question_id == pending["id"]
            if (
                not question_id
                and "?" in dialogue
                and status != "learner_question"
                and progress["turn"] < MAX_TURNS
            ):
                raise ValueError(
                    "Do not ask an untracked concern; select a fresh question ID or answer without a question"
                )
            if repeats_pending and (
                progress["must_advance"] or status not in {"unclear", "unsafe"}
            ):
                progress["must_advance"] = True
                raise ValueError(
                    "Accept the explanation and choose a fresh concern; do not ask for confirmation again"
                )
            if (
                not question_id
                and available
                and progress["turn"] < MAX_TURNS - 2
                and status in {"addressed", "partial"}
                and not result.get("ready_to_close")
            ):
                raise ValueError(
                    "Move to an available new concern in this reply instead of only reflecting"
                )
            break
        except ValueError as error:
            if progress["turn"] >= MAX_TURNS:
                return SUPPORT_CLOSING, True, debug(
                    scenario, progress, "turn_limit", complete=True
                )
            if attempt:
                raise
            logger.info("Repairing family dialogue: %s", error)
            progress["repair_reason"] = str(error)

    progress.pop("repair_reason", None)
    bank = {q["id"]: q for q in question_bank(scenario)}
    if scenario.opening_line:
        bank["opening"] = {"id": "opening", "theme": 0, "text": scenario.opening_line}

    reported = result.get("addressed_question_ids", [])
    if not isinstance(reported, list):
        raise ValueError("Invalid addressed concerns")
    question_id = result.get("question_id", "")
    allowed = {q["id"] for q in available}
    if (
        pending
        and not progress["must_advance"]
        and pending["id"] not in progress["addressed"]
    ):
        allowed.add(pending["id"])
    if question_id and question_id not in allowed:
        raise ValueError("Model selected an unavailable concern")

    addressed = set(progress["addressed"])
    if status not in {"unclear", "unsafe"}:
        addressed.update(q for q in reported if isinstance(q, str) and q in bank)
    if pending and status == "addressed":
        addressed.add(pending["id"])

    if question_id:
        addressed.discard(question_id)
        if question_id in {"opening", "0:0"}:
            addressed.difference_update({"opening", "0:0"})
    if "opening" in addressed or "0:0" in addressed:
        addressed.update({"opening", "0:0"} & bank.keys())
    if (
        pending
        and status != "addressed"
        and pending["id"] not in addressed
        and pending["id"] not in progress["unresolved"]
    ):
        progress["unresolved"].append(pending["id"])
    progress["addressed"] = sorted(addressed)
    progress["unresolved"] = [q for q in progress["unresolved"] if q not in addressed]

    covered = all(
        sum(
            q["theme"] == i and q["id"] != "opening" and q["id"] in addressed
            for q in bank.values()
        ) >= min(2, len(topic.possible_expressions))
        for i, topic in enumerate(scenario.topics)
    )
    latest = next((m.content for m in reversed(messages) if m.sender == "student"), "")
    evidence = result.get("readiness_evidence", "")
    readiness_checked = (
        isinstance(evidence, str)
        and bool(evidence.strip())
        and evidence.strip().casefold() in latest.casefold()
    )
    ready = (
        result.get("ready_to_close") is True
        and covered
        and not progress["unresolved"]
        and readiness_checked
        and not question_id
        and status != "unsafe"
    )
    if ready:
        progress["pending"] = None
        return scenario.closing or "I think I'm ready to go ahead now.", True, debug(
            scenario, progress, "ready", complete=True, ready=True
        )
    if progress["turn"] >= MAX_TURNS:
        statements = re.split(r"(?<=[.!?])\s+", dialogue)
        dialogue = " ".join(s for s in statements if "?" not in s)
        already_pausing = re.search(
            r"\b(?:i(?:'m| am) (?:still )?not ready|i need (?:a little |some |more )?(?:time|a moment|to pause))",
            dialogue.lower().replace("’", "'"),
        )
        closing = dialogue if already_pausing else f"{dialogue} {SUPPORT_CLOSING}".strip()
        return closing, True, debug(scenario, progress, "turn_limit", complete=True)

    question = next((q for q in available if q["id"] == question_id), None)
    if question:
        dialogue = render_question(dialogue, question, status, progress["turn"])
        progress["asked"].append(question)
        progress["pending"] = question
        progress["active_theme"] = max(progress["active_theme"], question["theme"])
    elif (
        pending
        and pending["id"] not in addressed
        and not progress["must_advance"]
    ):
        if question_id == pending["id"]:
            progress["follow_ups"][pending["id"]] = (
                progress["follow_ups"].get(pending["id"], 0) + 1
            )
    else:
        progress["pending"] = None
    return dialogue, False, debug(scenario, progress, "scenario_dialogue")


def debug(scenario, progress, reason, complete=False, ready=False):
    pending = progress["pending"]
    counts = {
        topic.id: sum(item["theme"] == i for item in progress["asked"])
        for i, topic in enumerate(scenario.topics)
    }
    addressed = set(progress.get("addressed", []))
    covered = [
        topic.id
        for i, topic in enumerate(scenario.topics)
        if any(
            q["theme"] == i and q["id"] in addressed
            for q in question_bank(scenario)
        )
    ]
    stage = (
        "ending"
        if complete or progress.get("turn", 0) >= MAX_TURNS - 2
        else "beginning" if progress.get("active_theme", 0) == 0 else "middle"
    )
    return {
        "core_question_state": progress,
        "current_stage": stage,
        "stage": stage,
        "reason": reason,
        "completion_status": complete,
        "voice_metadata": scenario.voice_metadata(),
        "active_topic": scenario.topics[pending["theme"]].id if pending else "",
        "covered_topics": covered,
        "unresolved_topics": [topic.id for topic in scenario.topics if topic.id not in covered],
        "topic_turn_counts": counts,
        "ending_ready": ready,
    }
