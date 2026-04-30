from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List
import json
import re

from openai import OpenAI


PROMPT_TEMPLATE_FILE = Path("prompts/prompt_template.md")
DEFAULT_ROLE_FILE = Path("prompts/role_prompt.md")


class ConversationPhase(str, Enum):
    INTRO = "intro"
    BEGINNING = "beginning"
    MIDDLE = "middle"
    ENDING = "ending"
    CLOSING = "closing"
    ENDED = "ended"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class RoleSections:
    role_summary: str
    introduction: str
    opening_line: str
    beginning: str
    middle: str
    ending: str
    closing: str
    meta_instructions: str
    begin_to_middle_cues: List[str]
    middle_to_ending_cues: List[str]
    voice_gender: str
    voice_style: str
    intro_voice_gender: str
    intro_voice_style: str


@dataclass
class ConversationState:
    phase: ConversationPhase = ConversationPhase.INTRO
    turn_count: int = 0
    history: List[Message] = field(default_factory=list)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _split_markdown_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for line in text.splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = _normalize_heading(match.group(1))
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _normalize_heading(text: str) -> str:
    value = re.sub(r"\(optional\)", "", (text or ""), flags=re.IGNORECASE)
    value = value.strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _section_by_prefix(sections: Dict[str, str], prefix: str) -> str:
    prefix = _normalize_heading(prefix)
    if prefix in sections:
        return sections[prefix]

    best_value = ""
    best_score = None
    prefix_tokens = prefix.split()
    for key, value in sections.items():
        key_tokens = key.split()
        if len(key_tokens) < len(prefix_tokens):
            continue
        if key_tokens[: len(prefix_tokens)] != prefix_tokens:
            continue
        extra_tokens = key_tokens[len(prefix_tokens) :]
        penalty = 5 if ("voice" in extra_tokens and "voice" not in prefix_tokens) else 0
        score = len(extra_tokens) + penalty
        if best_score is None or score < best_score:
            best_score = score
            best_value = value
    return best_value


def _section_by_aliases(sections: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        value = _section_by_prefix(sections, alias)
        if value:
            return value
    return ""


def _extract_quoted_line(text: str, prefer_last: bool = False) -> str:
    matches = re.findall(r'"([^"\n]+)"', text)
    if not matches:
        return ""
    return matches[-1].strip() if prefer_last else matches[0].strip()


def _extract_intro_script(introduction_text: str) -> str:
    if not introduction_text:
        return ""
    lines = [line.strip() for line in introduction_text.splitlines()]
    marker_idx = -1
    for i, line in enumerate(lines):
        if "copy everything" in line.lower():
            marker_idx = i
            break

    search = lines[marker_idx + 1 :] if marker_idx >= 0 else lines
    for line in search:
        if not line:
            continue
        if line.startswith("(") or line.startswith("["):
            continue
        return line
    return ""


def _extract_template_prefix(prompt_template_text: str) -> str:
    if not prompt_template_text:
        return ""
    if "{role}" in prompt_template_text:
        return prompt_template_text.split("{role}", 1)[0].strip()
    return prompt_template_text.strip()


def _infer_voice_gender(text: str) -> str:
    lower = (text or "").lower()
    if "male voice" in lower:
        return "male"
    if "female voice" in lower:
        return "female"
    return "female"


def _parse_cue_list(text: str) -> List[str]:
    if not text:
        return []
    cues: List[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
            cues.extend(parts)
        else:
            cues.append(line)
    return [cue.lower() for cue in cues if cue]


def parse_role_sections(role_prompt_text: str, prompt_template_text: str) -> RoleSections:
    parsed_json = None
    stripped = (role_prompt_text or "").strip()
    if stripped.startswith("{"):
        try:
            parsed_json = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_json = None

    if parsed_json and isinstance(parsed_json, dict):
        role_summary = str(parsed_json.get("role", "")).strip()
        intro = str(parsed_json.get("introduction", "")).strip()
        opening_line = str(parsed_json.get("opening_line", "")).strip()
        beginning = str(parsed_json.get("beginning", "")).strip()
        middle = str(parsed_json.get("middle", "")).strip()
        ending = str(parsed_json.get("end", "") or parsed_json.get("ending", "")).strip()
        closing = str(parsed_json.get("closing", "")).strip()
        meta_instructions = str(parsed_json.get("meta_instructions", "")).strip()
        begin_to_middle_cues = [str(x).lower() for x in parsed_json.get("begin_to_middle_cues", [])]
        middle_to_ending_cues = [str(x).lower() for x in parsed_json.get("middle_to_ending_cues", [])]
        voice_gender = str(parsed_json.get("voice_gender", "")).strip().lower()
        voice_style = str(parsed_json.get("voice_style", "")).strip()
        intro_voice_gender = str(parsed_json.get("intro_voice_gender", "")).strip().lower()
        intro_voice_style = str(parsed_json.get("intro_voice_style", "")).strip()
    else:
        role_sections = _split_markdown_sections(role_prompt_text)
        template_sections = _split_markdown_sections(prompt_template_text)

        role_summary = _section_by_aliases(role_sections, ["role", "role summary", "character"])
        intro = _section_by_aliases(role_sections, ["introduction", "introduction: greeting", "introduction: interview greeting"])
        opening_line = _section_by_aliases(role_sections, ["opening line", "beginning opening line"])
        beginning = _section_by_aliases(
            role_sections,
            ["beginning", "conversation progression: beginning", "conversation - beginning"],
        )
        middle = _section_by_aliases(
            role_sections,
            ["middle", "conversation progression: middle", "conversation - middle"],
        )
        ending = _section_by_aliases(
            role_sections,
            ["end", "ending", "conversation progression: end", "conversation progression: ending"],
        )
        closing = _section_by_aliases(role_sections, ["closing", "final response", "closing remark"])
        meta_instructions = _section_by_aliases(
            role_sections,
            ["meta instructions", "meta-instructions", "meta instruction", "notes", "constraints", "rules"],
        )
        begin_to_middle_cues = _parse_cue_list(
            _section_by_aliases(
                role_sections,
                ["beginning to middle cues", "begin-to-middle cues", "middle trigger", "middle triggers", "trigger"],
            )
        )
        middle_to_ending_cues = _parse_cue_list(
            _section_by_aliases(
                role_sections,
                ["middle to ending cues", "middle-to-ending cues", "ending trigger", "ending triggers"],
            )
        )
        voice_gender = _section_by_aliases(role_sections, ["voice gender", "voice"]).strip().lower()
        voice_style = _section_by_aliases(role_sections, ["voice style", "voice instructions"]).strip()
        intro_voice_gender = _section_by_aliases(
            role_sections,
            ["introduction voice gender", "intro voice gender"],
        ).strip().lower()
        intro_voice_style = _section_by_aliases(
            role_sections,
            ["introduction voice style", "intro voice style"],
        ).strip()

        if not closing:
            closing = _section_by_prefix(template_sections, "post-conversation protocol")

        intro_script = _extract_intro_script(intro)
        if intro_script:
            intro = intro_script

    if not intro:
        intro = "Greet the student nurse in character and briefly explain why you are here."
    if not role_summary:
        role_summary = role_prompt_text.strip() or "You are a standardized patient roleplay character."
    if not beginning:
        beginning = "Start with short answers. Do not volunteer too many details at first."
    if not middle:
        middle = "Open up more as trust is built. Reveal practical challenges when asked."
    if not ending:
        ending = "Become reflective and prepare to conclude if the nurse starts wrapping up."

    # If closing section contains lots of guidance text, try extracting a literal closing quote first.
    literal_closing = _extract_quoted_line(closing, prefer_last=True) if closing else ""
    if literal_closing and len(literal_closing) > 20:
        closing = literal_closing
    if not closing:
        closing = _extract_quoted_line(ending, prefer_last=True)
    if not closing:
        closing = (
            "Thank you for engaging with virtual conversation simulation. "
            "Please remember to download your conversation record."
        )
    if not meta_instructions:
        meta_instructions = (
            "Do not play both sides. Stay in character. "
            "Do not restart the introduction once conversation has begun."
        )
    if voice_gender not in {"male", "female"}:
        voice_gender = _infer_voice_gender(f"{role_prompt_text}\n{intro}\n{closing}")
    if not voice_style:
        voice_style = "speak naturally, clearly, and at a moderate pace"
    if intro_voice_gender not in {"male", "female"}:
        intro_voice_gender = voice_gender
    if not intro_voice_style:
        intro_voice_style = voice_style

    return RoleSections(
        role_summary=role_summary,
        introduction=intro,
        opening_line=opening_line,
        beginning=beginning,
        middle=middle,
        ending=ending,
        closing=closing,
        meta_instructions=meta_instructions,
        begin_to_middle_cues=begin_to_middle_cues,
        middle_to_ending_cues=middle_to_ending_cues,
        voice_gender=voice_gender,
        voice_style=voice_style,
        intro_voice_gender=intro_voice_gender,
        intro_voice_style=intro_voice_style,
    )


class StatefulVIPSON:
    """
    Standalone stateful backend (not integrated with Django yet).

    Rigid conversation flow:
    INTRO -> BEGINNING -> MIDDLE -> ENDING -> CLOSING -> ENDED
    """

    def __init__(
        self,
        api_key: str,
        role_file: Path = DEFAULT_ROLE_FILE,
        prompt_template_file: Path = PROMPT_TEMPLATE_FILE,
        model: str = "gpt-4o-mini",
        max_roleplay_turns: int = 18,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_roleplay_turns = max_roleplay_turns

        self.prompt_template_text = _read_text(prompt_template_file)
        self.role_prompt_text = _read_text(role_file)
        self.template_prefix = _extract_template_prefix(self.prompt_template_text)
        self.sections = parse_role_sections(self.role_prompt_text, self.prompt_template_text)

        self.state = ConversationState()

    def _history_text(self) -> str:
        lines = []
        for m in self.state.history:
            speaker = "Student" if m.role == "user" else "Assistant"
            lines.append(f"{speaker}: {m.content}")
        return "\n".join(lines).strip()

    def _llm(self, system_text: str, user_text: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        return (response.output_text or "").strip()

    def _enforce_voice_format(self, text: str) -> str:
        return self._enforce_voice_format_with(text, self.sections.voice_gender, self.sections.voice_style)

    def _enforce_voice_format_with(self, text: str, voice_gender: str, voice_style: str) -> str:
        text = (text or "").strip()
        if not text:
            return f"[{voice_gender} voice, {voice_style}]"

        if "[" not in text or "]" not in text:
            return f"{text}\n\n[{voice_gender} voice, {voice_style}]"

        # Ensure bracket contains male/female voice label.
        bracket_matches = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
        if not bracket_matches:
            return f"{text}\n\n[{voice_gender} voice, {voice_style}]"

        last = bracket_matches[-1]
        lower_last = last.lower()
        if "male voice" not in lower_last and "female voice" not in lower_last:
            fixed = f"{voice_gender} voice, {last.strip()}"
            return re.sub(r"\[([^\]]+)\]\s*$", f"[{fixed}]", text, count=1, flags=re.DOTALL)
        return text

    def _should_close(self, user_input: str) -> bool:
        lower = user_input.lower()
        return any(
            phrase in lower
            for phrase in [
                "thank you",
                "thanks",
                "goodbye",
                "bye",
                "that is all",
                "no that's all",
                "no, that's all",
                "nothing else",
                "we're done",
                "that covers everything",
            ]
        )

    def _should_move_to_middle(self, user_input: str) -> bool:
        lower = user_input.lower()
        cues = self.sections.begin_to_middle_cues or [
            "home",
            "medication",
            "medicine",
            "daily",
            "routine",
            "help",
            "support",
            "live",
            "safety",
            "fall",
            "transport",
            "pharmacy",
            "money",
            "family",
            "caregiver",
            "hospice",
            "feeding",
            "hydration",
            "tube",
            "pain",
            "comfort",
            "prognosis",
        ]
        return self.state.turn_count >= 2 or any(cue in lower for cue in cues)

    def _should_move_to_ending(self, user_input: str) -> bool:
        lower = user_input.lower()
        cues = self.sections.middle_to_ending_cues or [
            "anything else",
            "final question",
            "before we finish",
            "to summarize",
            "wrap up",
            "closing",
            "thank you",
            "i understand",
            "that helps",
            "i feel better",
        ]
        return self.state.turn_count >= 8 or any(cue in lower for cue in cues)

    def _advance_phase(self, user_input: str) -> None:
        if self.state.phase in {ConversationPhase.BEGINNING, ConversationPhase.MIDDLE, ConversationPhase.ENDING}:
            if self._should_close(user_input) or self.state.turn_count >= self.max_roleplay_turns:
                self.state.phase = ConversationPhase.CLOSING
                return

        if self.state.phase == ConversationPhase.BEGINNING and self._should_move_to_middle(user_input):
            self.state.phase = ConversationPhase.MIDDLE
            return

        if self.state.phase == ConversationPhase.MIDDLE and self._should_move_to_ending(user_input):
            self.state.phase = ConversationPhase.ENDING
            return

    def _stage_instructions(self) -> str:
        if self.state.phase == ConversationPhase.BEGINNING:
            return self.sections.beginning
        if self.state.phase == ConversationPhase.MIDDLE:
            return self.sections.middle
        if self.state.phase == ConversationPhase.ENDING:
            return self.sections.ending
        return self.sections.middle

    def start(self) -> str:
        if self.state.phase != ConversationPhase.INTRO:
            return "Conversation already started."

        assistant = self.sections.introduction
        if not assistant:
            system_text = (
                f"{self.template_prefix}\n\n"
                "You are a standardized roleplay participant assistant.\n"
                "Respond ONLY as the character and follow instructions exactly.\n"
                "Do not end the conversation here.\n"
                "Output format must always be:\n"
                "Dialogue\n\n"
                f"[{self.sections.voice_gender} voice, style instructions]"
            )
            user_text = (
                "Produce the fixed opening/introduction now.\n\n"
                f"Character introduction instructions:\n{self.sections.introduction}"
            )
            assistant = self._enforce_voice_format_with(
                self._llm(system_text, user_text),
                self.sections.intro_voice_gender,
                self.sections.intro_voice_style,
            )
        else:
            assistant = self._enforce_voice_format_with(
                assistant,
                self.sections.intro_voice_gender,
                self.sections.intro_voice_style,
            )

        self.state.history.append(Message(role="assistant", content=assistant))
        self.state.phase = ConversationPhase.BEGINNING
        return assistant

    def step(self, user_input: str) -> str:
        if self.state.phase == ConversationPhase.ENDED:
            return "This conversation is already closed."

        user_input = (user_input or "").strip()
        if not user_input:
            return "Please provide a message."

        self.state.history.append(Message(role="user", content=user_input))

        if self.state.phase in {ConversationPhase.BEGINNING, ConversationPhase.MIDDLE, ConversationPhase.ENDING}:
            self._advance_phase(user_input)
            if self.state.phase == ConversationPhase.CLOSING:
                pass
            else:
                # First response in beginning can be forced to exact opening line, if provided.
                if self.state.phase == ConversationPhase.BEGINNING and self.state.turn_count == 0 and self.sections.opening_line:
                    assistant = self._enforce_voice_format(self.sections.opening_line)
                else:
                    system_text = (
                        f"{self.template_prefix}\n\n"
                        "You are a standardized roleplay participant in a strict state-machine flow.\n"
                        "Current stage is fixed by the backend. Follow only this stage.\n"
                        "Do not restart or repeat any opening script.\n"
                        "Do not play both sides of the conversation.\n"
                        "Keep responses concise and natural.\n"
                        f"Global meta instructions:\n{self.sections.meta_instructions}\n"
                        "Output format must always be:\n"
                        "Dialogue\n\n"
                        f"[{self.sections.voice_gender} voice, style instructions]"
                    )
                    user_text = (
                        "Respond as the patient for the current stage.\n\n"
                        f"Character profile:\n{self.sections.role_summary}\n\n"
                        f"Stage: {self.state.phase.value}\n"
                        f"Stage instructions:\n{self._stage_instructions()}\n\n"
                        f"Conversation history:\n{self._history_text()}"
                    )
                    assistant = self._enforce_voice_format(self._llm(system_text, user_text))

                self.state.history.append(Message(role="assistant", content=assistant))
                self.state.turn_count += 1
                return assistant

        if self.state.phase == ConversationPhase.CLOSING:
            assistant = self._enforce_voice_format(self.sections.closing.strip())
            if not assistant:
                system_text = (
                    f"{self.template_prefix}\n\n"
                    "You are closing the roleplay.\n"
                    "Give a final closing remark and do not continue beyond that.\n"
                    "Output format must always be:\n"
                    "Dialogue\n\n"
                    f"[{self.sections.voice_gender} voice, style instructions]"
                )
                user_text = (
                    "Produce the final closing statement now.\n\n"
                    f"Closing instructions:\n{self.sections.closing}"
                )
                assistant = self._enforce_voice_format(self._llm(system_text, user_text))
            self.state.history.append(Message(role="assistant", content=assistant))
            self.state.phase = ConversationPhase.ENDED
            return assistant

        return "Unable to process the current state."


if __name__ == "__main__":
    import os

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    bot = StatefulVIPSON(api_key=api_key)
    print("=== Stateful VIPSON demo ===")
    print("Bot intro:")
    print(bot.start())
    print()
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        reply = bot.step(user)
        print(f"Bot: {reply}\n")
