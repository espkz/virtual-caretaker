#!/usr/bin/env python3
"""Interactive smoke tester for the production ConversationEngine.

This module intentionally owns only terminal I/O and transcript serialization.
Natural-language behavior remains in vip.conversation_engine.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
APP_ROOT = REPO_ROOT / "vipdjango"
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(APP_ROOT))

from vip.conversation_engine import ConversationEngine, format_voice_metadata  # noqa: E402


@dataclass
class Message:
    sender: str
    content: str
    voice_metadata: str = ""


def scenario_files():
    return sorted(
        path for path in (REPO_ROOT / "prompts").glob("role_*.md")
        if path.name != "role_prompt.md"
    )


def load_scenario(path):
    return Path(path).read_text(encoding="utf-8")


def choose_scenario(files, input_fn=input, output_fn=print):
    if not files:
        raise FileNotFoundError("No *_prompt.md scenarios found in prompts/.")
    output_fn("Available scenarios:")
    for index, path in enumerate(files, 1):
        output_fn(f"  {index}. {path.stem}")
    answer = input_fn("Select scenario [1]: ").strip() or "1"
    try:
        return files[int(answer) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("Choose one of the listed scenario numbers.") from exc


def transcript_payload(scenario_path, messages, debug, saved_at=None):
    return {
        "scenario": scenario_path.stem,
        "scenario_file": str(scenario_path),
        "saved_at": saved_at or datetime.now(timezone.utc).isoformat(),
        "completion_state": {
            "complete": bool(debug.get("completion_status", False)),
            "stage": debug.get("stage") or debug.get("current_stage", "beginning"),
            "turn_count": sum(m.sender == "student" for m in messages),
        },
        "messages": [
            {
                "turn": i,
                "sender": m.sender,
                "content": m.content,
                "voice_metadata": m.voice_metadata,
            }
            for i, m in enumerate(messages, 1)
        ],
    }


def save_transcript(directory, scenario_path, messages, debug):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = directory / f"{scenario_path.stem}_{stamp}.json"
    target.write_text(json.dumps(transcript_payload(scenario_path, messages, debug), indent=2), encoding="utf-8")
    return target


def run(args, input_fn=input, output_fn=print, engine=None):
    files = scenario_files()
    scenario_path = Path(args.scenario) if args.scenario else choose_scenario(files, input_fn, output_fn)
    role_text = load_scenario(scenario_path)
    if engine is None:
        api_key_path = REPO_ROOT / "api_key.txt"
        key = args.api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not key and api_key_path.exists():
            key = api_key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for the live tester.")
        global_prompt = (REPO_ROOT / "prompts" / "global_prompt.md").read_text(encoding="utf-8")
        engine = ConversationEngine(global_prompt, key)
    messages = []
    debug = {"stage": "beginning", "completion_status": False}
    saved = False

    def save():
        nonlocal saved
        path = save_transcript(Path(args.output_dir), scenario_path, messages, debug)
        saved = True
        output_fn(f"Saved transcript: {path}")
        return path

    output_fn(f"Scenario: {scenario_path.stem}")
    character, complete, info = engine.respond(role_text, messages)
    debug.update(info or {}, completion_status=complete)
    if character:
        voice = info.get("voice_metadata", "") if info else ""
        messages.append(Message("assistant", character, voice))
        output_fn(f"\nCHARACTER:\n{character}\n\n{format_voice_metadata(voice)}" if voice else f"\nCHARACTER:\n{character}")
    while not complete:
        try:
            learner = input_fn("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nStopping.")
            if messages and not saved:
                save()
            return messages
        if learner == "/help":
            output_fn("/save  save transcript\n/stop  stop and offer to save\n/status  show state\n/help  show commands")
            continue
        if learner == "/status":
            output_fn(f"Scenario: {scenario_path.stem}\nStage: {debug.get('stage')}\nTurn count: {sum(m.sender == 'student' for m in messages)}\nComplete: {debug.get('completion_status', False)}")
            continue
        if learner == "/save":
            save()
            continue
        if learner == "/stop":
            if not saved and input_fn("Save transcript? [Y/n] ").strip().lower() not in {"n", "no"}:
                save()
            return messages
        if not learner:
            continue
        messages.append(Message("student", learner))
        saved = False
        try:
            character, complete, info = engine.respond(role_text, messages, conversation_state=debug)
        except Exception as exc:
            messages.pop()
            output_fn(f"Response unavailable ({type(exc).__name__}). Please try your reply again.")
            continue
        debug.update(info or {}, completion_status=complete)
        if args.verbose:
            output_fn(json.dumps({"history": [m.__dict__ for m in messages], "debug": debug}, indent=2, default=str))
        if character:
            voice = info.get("voice_metadata", "") if info else ""
            messages.append(Message("assistant", character, voice))
            output_fn(f"\nCHARACTER:\n{character}\n\n{format_voice_metadata(voice)}" if voice else f"\nCHARACTER:\n{character}")
        if complete:
            output_fn("\nConversation complete.")
            if not saved:
                save()
    return messages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="path to an existing *_prompt.md scenario")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "test_conversations"))
    parser.add_argument("--api-key")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
