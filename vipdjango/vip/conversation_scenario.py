from dataclasses import asdict, dataclass

from .prompt_utils import find_section_by_aliases, split_markdown_sections


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

    def to_state(self):
        return asdict(self)

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
        learner=section(["learner role", "user role"]) or "nursing student",
        voice_gender=gender,
        voice_style=voice_style,
        introduction=section(["introduction", "introduction: greeting"]),
        opening_line=section(["opening line"]),
        beginning=section(["beginning", "conversation progression: beginning"]),
        middle=section(["middle", "conversation progression: middle"]),
        ending=section(["ending", "end", "conversation progression: end", "conversation progression: ending"]),
        closing=section(["closing", "final response"]),
        meta=section(["meta instructions", "meta instruction", "meta-instructions", "notes"]),
        beginning_to_middle_cues=section(["beginning to middle cues", "begin-to-middle cues", "middle triggers"]),
        middle_to_ending_cues=section(["middle to ending cues", "middle-to-ending cues", "ending triggers"]),
        end_of_conversation_cues=section(["end of conversation cues"]),
    )
