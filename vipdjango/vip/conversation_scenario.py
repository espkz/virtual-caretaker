import re
from dataclasses import asdict, dataclass

from .prompt_utils import find_section_by_aliases, split_markdown_sections


DEFAULT_LEARNER_ROLE = "the learner in this simulation"


@dataclass(frozen=True)
class ScenarioObjective:
    """A scenario concern expressed as guidance, not a required script line."""

    id: str
    description: str
    possible_expressions: str
    resolved_when: str


@dataclass(frozen=True)
class Scenario:
    character: str
    learner: str
    voice_gender: str
    voice_style: str
    introduction: str
    opening_line: str
    beginning: str
    middle: str
    ending: str
    closing: str
    meta: str
    beginning_to_middle_cues: str
    middle_to_ending_cues: str
    end_of_conversation_cues: str
    objectives: tuple[ScenarioObjective, ...] = ()
    background_context: str = ""

    def to_state(self):
        state = asdict(self)
        state["objectives"] = [asdict(objective) for objective in self.objectives]
        return state

    @classmethod
    def from_state(cls, state):
        values = dict(state)
        values["objectives"] = tuple(
            objective
            if isinstance(objective, ScenarioObjective)
            else ScenarioObjective(**objective)
            for objective in values.get("objectives", ())
        )
        return cls(**values)

    def voice_metadata(self):
        return f"{self.voice_gender} voice, {self.voice_style}"


def parse_scenario_prompt(role_text: str) -> Scenario:
    sections = split_markdown_sections(role_text)

    def section(aliases):
        return find_section_by_aliases(sections, aliases).strip()

    gender = (section(["voice gender", "voice"]) or "female").lower()
    if gender not in {"male", "female"}:
        gender = "female"
    voice_style = section(["voice style", "voice instructions"]) or "speak naturally and clearly"
    return Scenario(
        character=section(["role", "role summary", "character"]) or role_text.strip(),
        background_context=section(["background and context", "background", "context"]),
        learner=section(["learner role", "user role"]) or DEFAULT_LEARNER_ROLE,
        voice_gender=gender,
        voice_style=voice_style,
        introduction=section(["introduction", "introduction: greeting"]),
        opening_line=section(["opening line"]),
        beginning=section(["beginning", "conversation progression: beginning"]),
        middle=section(["middle", "conversation progression: middle"]),
        ending=section(["ending", "end", "conversation progression: end", "conversation progression: ending"]),
        closing=section(["closing", "final response"]),
        meta=section(["meta instructions", "meta instruction", "meta-instructions", "notes"]),
        beginning_to_middle_cues=section([
            "beginning to middle cues",
            "beginning to middle transition",
            "begin-to-middle cues",
            "middle triggers",
        ]),
        middle_to_ending_cues=section([
            "middle to ending cues",
            "middle to ending transition",
            "middle-to-ending cues",
            "ending triggers",
        ]),
        end_of_conversation_cues=section(["end of conversation cues"]),
        objectives=parse_scenario_objectives(
            section(["conversation objectives", "conversation goals", "objectives", "goals"])
        ),
    )


_OBJECTIVE_HEADING_RE = re.compile(
    r"^\s*(?:#{3,6}\s+|OBJECTIVE\s*:\s*)(.+?)\s*$",
    flags=re.IGNORECASE,
)
_OBJECTIVE_FIELD_RE = re.compile(
    r"^\s*(possible expressions?|possible concerns?|example questions?|resolved when|resolution conditions?)\s*:\s*(.*)$",
    flags=re.IGNORECASE,
)


def _objective_id(title, used_ids, index):
    value = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    value = value or f"objective-{index}"
    candidate = value
    suffix = 2
    while candidate in used_ids:
        candidate = f"{value}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _parse_objective_block(title, lines, used_ids, index):
    sections = {"description": [], "possible_expressions": [], "resolved_when": []}
    current = "description"
    for line in lines:
        value = line.strip()
        if not value:
            continue
        field = _OBJECTIVE_FIELD_RE.match(value)
        if field:
            field_name = field.group(1).lower()
            if field_name.startswith(("possible", "example")):
                current = "possible_expressions"
            else:
                current = "resolved_when"
            if field.group(2).strip():
                sections[current].append(field.group(2).strip())
            continue
        sections[current].append(re.sub(r"^[-*+]\s+", "", value))

    description = " ".join(sections["description"]).strip() or title.strip()
    return ScenarioObjective(
        id=_objective_id(title, used_ids, index),
        description=description,
        possible_expressions="\n".join(sections["possible_expressions"]).strip(),
        resolved_when=" ".join(sections["resolved_when"]).strip(),
    )


def parse_scenario_objectives(objectives_text: str) -> tuple[ScenarioObjective, ...]:
    """Parse optional generic objective blocks from a scenario prompt.

    Each ``###`` heading (or ``OBJECTIVE:`` line) starts one objective. The
    parser intentionally returns no objectives for older prompts without the
    optional section, preserving their existing behavior.
    """
    lines = (objectives_text or "").splitlines()
    blocks = []
    current_title = ""
    current_lines = []
    for line in lines:
        match = _OBJECTIVE_HEADING_RE.match(line)
        if match:
            if current_title:
                blocks.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = []
        elif current_title:
            current_lines.append(line)
    if current_title:
        blocks.append((current_title, current_lines))

    used_ids = set()
    return tuple(
        _parse_objective_block(title, block_lines, used_ids, index)
        for index, (title, block_lines) in enumerate(blocks, start=1)
        if title.strip()
    )
