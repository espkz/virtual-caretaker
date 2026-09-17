import re


def normalize_heading(text):
    value = re.sub(r"\(optional\)", "", (text or ""), flags=re.IGNORECASE)
    value = value.strip().lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_gender(value, default="female"):
    normalized = normalize_heading(value)
    if normalized in {"male", "female"}:
        return normalized
    if normalized.startswith("male "):
        return "male"
    if normalized.startswith("female "):
        return "female"
    return default


_UNSAFE_VOICE_STYLE_WORDS = {
    "assistant", "complete", "ending", "instruction", "learner", "prompt",
    "role", "stage", "state", "system", "tool", "transition", "user",
}


def normalize_voice_style(value, default="natural"):
    """Return safe comma-separated ElevenLabs delivery tags.

    Each tag may contain one or two words. Voice style is presentation
    metadata, so keeping each tag deliberately small also prevents
    scenario/state instructions from being smuggled into TTS tags.
    """
    raw_tags = [tag.strip().lower() for tag in (value or "").split(",") if tag.strip()]
    if not raw_tags:
        return default

    tags = []
    for raw_tag in raw_tags:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", raw_tag)
        normalized = " ".join(words)
        if (
            not 1 <= len(words) <= 2
            or normalized != re.sub(r"\s+", " ", raw_tag).strip()
            or any(word in _UNSAFE_VOICE_STYLE_WORDS for word in words)
        ):
            return default
        if normalized not in tags:
            tags.append(normalized)
    return ", ".join(tags)


def split_markdown_sections(text):
    sections = {}
    current = ""
    nested = ""
    for line in (text or "").splitlines():
        match = re.match(r"^\s*(#{2,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            heading = normalize_heading(match.group(2))
            if level == 2:
                current = heading
                nested = ""
                sections.setdefault(current, [])
            else:
                nested = heading
                sections.setdefault(nested, [])
                if current:
                    sections[current].append(line)
        elif current:
            sections[current].append(line)
            if nested:
                sections[nested].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def find_section_by_aliases(sections, aliases):
    for alias in aliases:
        normalized_alias = normalize_heading(alias)
        if normalized_alias in sections:
            return sections[normalized_alias]

    best_value = ""
    best_score = None
    for alias in aliases:
        alias_tokens = normalize_heading(alias).split()
        for key, value in sections.items():
            key_tokens = key.split()
            if key_tokens[: len(alias_tokens)] != alias_tokens:
                continue
            extra_tokens = key_tokens[len(alias_tokens) :]
            penalty = 5 if ("voice" in extra_tokens and "voice" not in alias_tokens) else 0
            score = len(extra_tokens) + penalty
            if best_score is None or score < best_score:
                best_score = score
                best_value = value
    return best_value


def find_exact_section_by_aliases(sections, aliases):
    """Return a section only when its heading exactly matches an alias."""
    for alias in aliases:
        normalized_alias = normalize_heading(alias)
        if normalized_alias in sections:
            return sections[normalized_alias]
    return ""


_REQUIRED_PROMPT_SECTIONS = (
    ("Simulation Mode", ("simulation mode",), False),
    ("Role", ("role", "role summary", "character"), False),
    ("Background and Context", ("background and context", "background", "context"), False),
    ("User Role", ("learner role", "user role"), False),
    ("Conversation Goals", ("conversation objectives", "conversation goals", "objectives", "goals"), False),
    ("Introduction", ("introduction", "introduction: greeting"), False),
    ("Opening Line", ("opening line",), False),
    ("Beginning", ("beginning", "conversation progression: beginning"), False),
    ("Beginning to Middle Transition", ("beginning to middle transition", "beginning to middle cues"), False),
    ("Middle", ("middle", "conversation progression: middle"), False),
    ("Middle to Ending Transition", ("middle to ending transition", "middle to ending cues"), False),
    ("Ending", ("ending", "end", "conversation progression: end", "conversation progression: ending"), False),
    ("Closing", ("closing", "final response"), False),
    ("Introduction Voice", ("introduction voice", "narrator voice"), True),
    ("Roleplay Voice", ("roleplay voice", "character voice"), True),
    ("Voice Style", ("voice style", "voice instructions"), True),
)


def missing_required_prompt_fields(title, content):
    """List editor fields that are empty or invalid in persisted Markdown."""
    missing = []
    if not (title or "").strip():
        missing.append("Title")
    sections = split_markdown_sections(content)
    for label, aliases, exact_only in _REQUIRED_PROMPT_SECTIONS:
        finder = find_exact_section_by_aliases if exact_only else find_section_by_aliases
        value = finder(sections, aliases).strip()
        if not value:
            missing.append(label)
        elif label == "Voice Style" and not normalize_voice_style(value, default=""):
            missing.append(label)
    return tuple(missing)


def is_prompt_complete(title, content):
    return not missing_required_prompt_fields(title, content)


def extract_fixed_section(text, aliases):
    """Extract author-provided fixed content, excluding template instructions.

    Scenario role prompts treat the whole section body as author content. Shared
    templates may wrap their author content in explicit fixed-content markers.
    """
    sections = split_markdown_sections(text)
    value = find_section_by_aliases(sections, aliases).strip()
    if not value:
        return ""
    start_marker = "<!-- fixed-content -->"
    end_marker = "<!-- /fixed-content -->"
    if start_marker not in value:
        return value
    content = value.split(start_marker, 1)[1]
    if end_marker in content:
        content = content.split(end_marker, 1)[0]
    return content.strip()
