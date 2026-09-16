"""Replay the four faculty transcripts against revised Rachel (explicit API opt-in).

Only nurse messages are replayed; Rachel replies are generated afresh. Later
nurse messages may no longer match her new questions, so this is a regression
probe, not an assessment of a fully interactive encounter.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vipdjango"))
from vip.conversation_engine import ConversationEngine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--session", choices=["56", "57", "58", "59"], action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("Configure OPENAI_API_KEY first.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for session in args.session or ["56", "57", "58", "59"]:
            number = 1 if session in {"56", "57"} else 2
            source = ROOT / "FW__AI-SP__AI_SP_is_ready_for_testing" / f"chat_session_{session} (1).txt"
            nurse_lines = re.findall(r"^\[[^\]]+\] Student: (.*)$", source.read_text(), re.M)
            role = (ROOT / "prompts" / f"role_rachel_ellison_{number}.md").read_text()
            engine = ConversationEngine("", key)
            state, messages = {}, []
            for turn, nurse in enumerate(nurse_lines, 1):
                messages.append(SimpleNamespace(sender="student", content=nurse))
                started = perf_counter()
                try:
                    text, complete, state = engine.respond(role, messages, state)
                except Exception as error:
                    output.write(json.dumps({"session": session, "turn": turn, "error": type(error).__name__}) + "\n")
                    output.flush()
                    raise
                seconds = round(perf_counter() - started, 2)
                messages.append(SimpleNamespace(sender="assistant", content=text))
                output.write(json.dumps({"session": session, "scenario": number, "model": engine.model,
                    "turn": turn, "nurse": nurse, "rachel": text, "complete": complete,
                    "seconds": seconds, "progress": state}, ensure_ascii=False) + "\n")
                output.flush()
                print(f"Session {session}, turn {turn}: {seconds}s, {state['reason']}", flush=True)
                if complete:
                    break


if __name__ == "__main__":
    main()
