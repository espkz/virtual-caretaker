"""Bounded practice with instructor-authored dialogue and model-selected reactions.

The model assesses the reply and selects a question ID. It cannot supply spoken
text, change roles, skip themes, or control termination. Coverage means practice
opportunity, never a clinical grade. State is committed with the assistant turn.
"""
from copy import deepcopy


REACTIONS = {
    "processing": "I'm still taking that in.",
    "heard": "Thank you for explaining that.",
    "worried": "I'm still worried about this.",
    "overwhelmed": "This is a lot to take in.",
}
CLARIFICATION = "I'm not sure I understand. Could you explain that concern in simpler terms?"
SUPPORT_CLOSING = "I need a little more time and support before we continue. I'd like to speak with someone from the care team."
PAUSE_CLOSING = "I need to pause here and take some time to process this."


def enabled(scenario):
    return scenario.simulation_mode == "roleplay" and bool(scenario.topics) and all(topic.possible_expressions for topic in scenario.topics)


def initial_state(scenario, messages):
    """Recover older transcripts conservatively without reopening spoken questions."""
    state = {"asked": [], "pending": None, "clarified": [], "unresolved": []}
    for message in messages:
        if message.sender != "assistant":
            continue
        text = message.content.strip()
        if scenario.opening_line and text == scenario.opening_line.strip():
            question = {"id": "opening", "theme": 0, "text": scenario.opening_line}
        else:
            question = next((
                {"id": f"{i}:{j}", "theme": i, "text": expression}
                for i, topic in enumerate(scenario.topics)
                for j, expression in enumerate(topic.possible_expressions)
                if expression in text
            ), None)
        if question and question["id"] not in [item["id"] for item in state["asked"]]:
            if state["pending"]:
                # No assessment was persisted for this older exchange.
                state["unresolved"].append(state["pending"]["id"])
            state["asked"].append(question)
            state["pending"] = question
    return state


def opening_state(scenario):
    question = {"id": "opening", "theme": 0, "text": scenario.opening_line}
    return {"asked": [question], "pending": question, "clarified": [], "unresolved": []}


def candidates(scenario, progress):
    asked_ids = {item["id"] for item in progress["asked"]}
    if "opening" in asked_ids:
        # The opening raises the first theme's first concern. Do not ask its
        # canonical equivalent again immediately after the learner answers it.
        asked_ids.add("0:0")
    for i, topic in enumerate(scenario.topics):
        count = sum(item["theme"] == i for item in progress["asked"])
        if count < min(2, len(topic.possible_expressions)):
            return [
                {"id": f"{i}:{j}", "theme": i, "text": expression}
                for j, expression in enumerate(topic.possible_expressions)
                if f"{i}:{j}" not in asked_ids
            ]
    return []


def respond(scenario, messages, persisted, assess):
    progress = deepcopy(persisted or initial_state(scenario, messages))
    available = candidates(scenario, progress)
    pending = progress["pending"]
    result = assess(scenario, messages, pending, available)
    status = result.get("answer_status")
    if status not in {"addressed", "unclear", "unsafe"}:
        raise ValueError("Invalid core-question assessment")

    # One repair opportunity per theme, even for repeatedly evasive responses.
    if pending and status != "addressed" and pending["theme"] not in progress["clarified"]:
        progress["clarified"].append(pending["theme"])
        return CLARIFICATION, False, debug(scenario, progress, "clarification")
    if pending and status != "addressed":
        progress["unresolved"].append(pending["id"])
    progress["pending"] = None

    if not available:
        dialogue = SUPPORT_CLOSING if progress["unresolved"] else (scenario.closing or PAUSE_CLOSING)
        return dialogue, True, debug(scenario, progress, "needs_support" if progress["unresolved"] else "core_questions_complete", complete=True)

    # Invalid, invented, or already-used IDs fall back to a real remaining question.
    question = next((q for q in available if q["id"] == result.get("question_id")), available[0])
    progress["asked"].append(question)
    progress["pending"] = question
    reaction = REACTIONS.get(result.get("reaction"), REACTIONS["processing"])
    if status != "addressed":
        reaction = REACTIONS["worried"]
    return f"{reaction} {question['text']}", False, debug(scenario, progress, "core_question")


def debug(scenario, progress, reason, complete=False):
    pending = progress["pending"]
    counts = {
        topic.id: sum(item["theme"] == i for item in progress["asked"])
        for i, topic in enumerate(scenario.topics)
    }
    covered = [
        topic.id for i, topic in enumerate(scenario.topics)
        if counts[topic.id] >= min(2, len(topic.possible_expressions))
        and (not pending or pending["theme"] != i)
    ]
    return {
        "core_question_state": progress,
        "current_stage": "ending" if complete else "middle",
        "stage": "ending" if complete else "middle",
        "reason": reason,
        "completion_status": complete,
        "voice_metadata": scenario.voice_metadata(),
        "active_topic": scenario.topics[pending["theme"]].id if pending else "",
        "covered_topics": covered,
        "unresolved_topics": [topic.id for topic in scenario.topics if topic.id not in covered],
        "topic_turn_counts": counts,
        "ending_ready": complete and not progress["unresolved"],
    }
