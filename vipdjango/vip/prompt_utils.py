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


def split_markdown_sections(text):
    sections = {}
    current = ""
    for line in (text or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = normalize_heading(match.group(1))
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
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
