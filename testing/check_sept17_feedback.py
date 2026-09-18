"""Opt-in live check of the session-62 follow-up and varied opening concerns."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vipdjango"))
from vip import core_questions  # noqa: E402
from vip.conversation_engine import ConversationEngine  # noqa: E402
from vip.conversation_scenario import parse_scenario_prompt  # noqa: E402


def message(sender, content):
    return SimpleNamespace(sender=sender, content=content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ["OPENAI_API_KEY"]
    role = (ROOT / "prompts/role_rachel_ellison_1.md").read_text()
    scenario = parse_scenario_prompt(role)
    engine = ConversationEngine("", key)
    concern = next(q for q in core_questions.question_bank(scenario) if q["id"] == "1:0")
    progress = core_questions.opening_state(scenario)
    progress["asked"].append(concern)
    progress["pending"] = concern
    progress["active_theme"] = 1
    progress["addressed"] = ["opening", "0:0"]
    history = [message("assistant", concern["text"])]
    state = {"core_question_state": progress}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for nurse in (
            "I think your caregiver team has likely tried what they think is best. Is there another treatment you are hoping for?",
            "I would expect so, yes.",
        ):
            history.append(message("student", nurse))
            reply, done, state = engine.respond(role, history, state)
            history.append(message("assistant", reply))
            out.write(json.dumps({"case": "specialist_followup", "nurse": nurse, "rachel": reply, "state": state}) + "\n")
            out.flush()
            print(reply, flush=True)
        if "1:0" not in state["core_question_state"]["addressed"]:
            raise AssertionError("Reasonable confirmation did not resolve the specialist concern")
        if (state["core_question_state"].get("pending") or {}).get("id") == "1:0":
            raise AssertionError("Repeated the specialist question")
        # Deliberately choose two distinct plans: verify the model actually uses
        # the available randomized choice rather than its favorite fixed opening.
        for first in ("0:2", "0:4"):
            plan = [first] + [q["id"] for q in core_questions.question_bank(scenario) if q["id"] != first]
            initial = {"asked": [], "pending": None, "clarified": [], "unresolved": [], "question_order": plan}
            reply, _, state = engine.respond(role, [message("student", "Hello, I'm your nurse today.")], {"core_question_state": initial})
            out.write(json.dumps({"case": "opening_variation", "expected_id": first, "rachel": reply, "state": state}) + "\n")
            out.flush()
            if state["core_question_state"]["pending"]["id"] != first:
                raise AssertionError("Model ignored the session's opening sample")
            print(reply, flush=True)
    print("Focused checks passed.")


if __name__ == "__main__":
    main()
