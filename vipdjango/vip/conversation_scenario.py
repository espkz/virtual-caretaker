import re
from dataclasses import asdict, dataclass

from .prompt_utils import (
    find_exact_section_by_aliases,
    find_section_by_aliases,
    normalize_voice_style,
    split_markdown_sections,
)


DEFAULT_LEARNER_ROLE = "the learner in this simulation"


@dataclass(frozen=True)
class ScenarioObjective:
    """A scenario concern expressed as guidance, not a required script line."""

    id: str
    description: str
    possible_expressions: str
    resolved_when: str


@dataclass(frozen=True)
class ScenarioTopic:
    """A concrete conversational concern nested under the scenario guidance."""

    id: str
    title: str
    possible_expressions: tuple[str, ...] = ()


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
    topics: tuple[ScenarioTopic, ...] = ()
    simulation_mode: str = "roleplay"
    introduction_voice_id: str = ""
    roleplay_voice_id: str = ""

    def to_state(self):
        state = asdict(self)
        state["objectives"] = [asdict(objective) for objective in self.objectives]
        return state

    @classmethod
    def from_state(cls, state):
        values = dict(state)
        values.setdefault("voice_gender", "")
        values.setdefault("introduction_voice_id", "")
        values.setdefault("roleplay_voice_id", "")
        values["objectives"] = tuple(
            objective
            if isinstance(objective, ScenarioObjective)
            else ScenarioObjective(**objective)
            for objective in values.get("objectives", ())
        )
        values["topics"] = tuple(
            topic
            if isinstance(topic, ScenarioTopic)
            else ScenarioTopic(
                id=topic.get("id", ""),
                title=topic.get("title", ""),
                possible_expressions=tuple(topic.get("possible_expressions", ())),
            )
            for topic in values.get("topics", ())
        )
        return cls(**values)

    def voice_metadata(self):
        return normalize_voice_style(self.voice_style)


def parse_scenario_prompt(role_text: str) -> Scenario:
    sections = split_markdown_sections(role_text)

    def section(aliases):
        return find_section_by_aliases(sections, aliases).strip()

    def exact_section(aliases):
        return find_exact_section_by_aliases(sections, aliases).strip()

    gender = (section(["voice gender", "voice"]) or "female").lower()
    if gender not in {"male", "female"}:
        gender = "female"
    voice_style = normalize_voice_style(section(["voice style", "voice instructions"]))
    middle = section(["middle", "conversation progression: middle"])
    return Scenario(
        simulation_mode=section(["simulation mode"]).lower() or "roleplay",
        character=section(["role", "role summary", "character"]) or role_text.strip(),
        background_context=section(["background and context", "background", "context"]),
        learner=section(["learner role", "user role"]) or DEFAULT_LEARNER_ROLE,
        voice_gender=gender,
        voice_style=voice_style,
        introduction_voice_id=exact_section(["introduction voice", "narrator voice"]),
        roleplay_voice_id=exact_section(["roleplay voice", "character voice"]),
        introduction=section(["introduction", "introduction: greeting"]),
        opening_line=section(["opening line"]),
        beginning=section(["beginning", "conversation progression: beginning"]),
        middle=middle,
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
        topics=parse_scenario_topics(middle),
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


_TOPIC_HEADING_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
_TOPIC_EXAMPLE_RE = re.compile(r"^\s+(?:\d+[.)]|[-*+])\s+(.+?)\s*$")


def _topic_id(title, used_ids, index):
    value = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    value = value or f"topic-{index}"
    candidate = value
    suffix = 2
    while candidate in used_ids:
        candidate = f"{value}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def parse_scenario_topics(middle_text: str) -> tuple[ScenarioTopic, ...]:
    """Extract concrete topic clusters from the existing Middle guidance.

    New prompts commonly use a top-level bullet for a topic and indented
    numbered examples beneath it. Older prompts often use one bullet per
    concern. Supporting both formats gives the application an explicit
    progress ledger without requiring a prompt migration or a rigid script.
    """
    blocks = []
    current_title = ""
    current_examples = []

    def flush():
        nonlocal current_title, current_examples
        if current_title.strip():
            blocks.append((current_title.strip(), tuple(current_examples)))
        current_title = ""
        current_examples = []

    for line in (middle_text or "").splitlines():
        heading = _TOPIC_HEADING_RE.match(line)
        if heading and not line.startswith((" ", "\t")):
            flush()
            current_title = heading.group(1).strip()
            continue
        if current_title:
            example = _TOPIC_EXAMPLE_RE.match(line)
            if example:
                current_examples.append(example.group(1).strip())

    flush()
    used_ids = set()
    return tuple(
        ScenarioTopic(
            id=_topic_id(title, used_ids, index),
            title=title,
            possible_expressions=tuple(
                example.strip().strip('"“”')
                for example in examples
                if example.strip()
            ),
        )
        for index, (title, examples) in enumerate(blocks, start=1)
    )
