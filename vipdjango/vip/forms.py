import re

from django import forms
from .models import RolePrompt


def _normalize_heading(text):
    value = re.sub(r"\(optional\)", "", (text or ""), flags=re.IGNORECASE)
    value = value.strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_gender(value, default="female"):
    v = _normalize_heading(value)
    if v in {"male", "female"}:
        return v
    if v.startswith("male "):
        return "male"
    if v.startswith("female "):
        return "female"
    return default


def _split_markdown_sections(text):
    sections = {}
    current = ""
    for line in (text or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = _normalize_heading(match.group(1))
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _find_section_by_aliases(sections, aliases):
    normalized = {key: value for key, value in sections.items()}

    # Pass 1: exact match only.
    for alias in aliases:
        alias = _normalize_heading(alias)
        if alias in normalized:
            return normalized[alias]

    # Pass 2: ranked prefix match (prefer non-voice variants).
    best_value = ""
    best_score = None
    for alias in aliases:
        alias = _normalize_heading(alias)
        alias_tokens = alias.split()
        for key, value in normalized.items():
            key_tokens = key.split()
            if len(key_tokens) < len(alias_tokens):
                continue
            if key_tokens[: len(alias_tokens)] != alias_tokens:
                continue
            extra_tokens = key_tokens[len(alias_tokens) :]
            penalty = 5 if ("voice" in extra_tokens and "voice" not in alias_tokens) else 0
            score = len(extra_tokens) + penalty
            if best_score is None or score < best_score:
                best_score = score
                best_value = value
    return best_value


class RolePromptForm(forms.Form):
    title = forms.CharField(max_length=120)
    is_active = forms.BooleanField(required=False)

    role = forms.CharField(widget=forms.Textarea(attrs={"rows": 5}))
    learner_role = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional: who is the learner/user in this simulation.",
    )
    voice_gender = forms.ChoiceField(choices=[("female", "female"), ("male", "male")], initial="female")
    voice_style = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="Optional speaking style. Example: warm, calm, relatively slow.",
    )
    intro_voice_gender = forms.ChoiceField(
        choices=[("female", "female"), ("male", "male")],
        initial="female",
        required=False,
        help_text="Optional. If blank, uses Voice Gender.",
    )
    intro_voice_style = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="Optional intro style. If blank, uses Voice Style.",
    )
    introduction = forms.CharField(widget=forms.Textarea(attrs={"rows": 5}))
    opening_line = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. Use this only if you want a fixed first in-character response.",
    )
    beginning = forms.CharField(widget=forms.Textarea(attrs={"rows": 6}))
    begin_to_middle_cues = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. One cue per line (bullet points are fine).",
    )
    middle = forms.CharField(widget=forms.Textarea(attrs={"rows": 6}))
    middle_to_ending_cues = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. One cue per line (bullet points are fine).",
    )
    ending = forms.CharField(widget=forms.Textarea(attrs={"rows": 6}))
    closing = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="Optional. If left blank, the default closing behavior will be used.",
    )
    meta_instructions = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="Optional global constraints.",
    )

    @classmethod
    def initial_from_prompt(cls, prompt):
        return cls.initial_from_content(
            prompt.content,
            title=prompt.title,
            is_active=prompt.is_active,
        )

    @classmethod
    def initial_from_content(cls, content, title="", is_active=False):
        sections = _split_markdown_sections(content)
        voice_gender = _normalize_gender(_find_section_by_aliases(sections, ["voice gender"]), default="female")
        role = _find_section_by_aliases(sections, ["role", "role summary", "character"])
        if not role:
            role = (content or "").strip()
        return {
            "title": title,
            "is_active": is_active,
            "role": role,
            "learner_role": _find_section_by_aliases(sections, ["learner role", "user role"]),
            "voice_gender": voice_gender,
            "voice_style": _find_section_by_aliases(sections, ["voice style", "voice instructions"]),
            "intro_voice_gender": _normalize_gender(
                _find_section_by_aliases(sections, ["introduction voice gender", "intro voice gender"]),
                default=voice_gender,
            ),
            "intro_voice_style": _find_section_by_aliases(sections, ["introduction voice style", "intro voice style"]),
            "introduction": _find_section_by_aliases(sections, ["introduction", "introduction: greeting"]),
            "opening_line": _find_section_by_aliases(sections, ["opening line"]),
            "beginning": _find_section_by_aliases(sections, ["beginning", "conversation progression: beginning"]),
            "begin_to_middle_cues": _find_section_by_aliases(
                sections,
                ["beginning to middle cues", "middle triggers", "middle trigger", "trigger"],
            ),
            "middle": _find_section_by_aliases(sections, ["middle", "conversation progression: middle"]),
            "middle_to_ending_cues": _find_section_by_aliases(
                sections,
                ["middle to ending cues", "ending triggers", "ending trigger"],
            ),
            "ending": _find_section_by_aliases(
                sections,
                ["ending", "end", "conversation progression: end", "conversation progression: ending"],
            ),
            "closing": _find_section_by_aliases(sections, ["closing", "final response"]),
            "meta_instructions": _find_section_by_aliases(
                sections,
                ["meta instructions", "meta-instructions", "meta instruction", "notes"],
            ),
        }

    def render_markdown_content(self):
        data = self.cleaned_data

        def block(header, value):
            value = (value or "").strip()
            return f"## {header}\n{value}\n"

        parts = [
            block("Role", data.get("role")),
            block("Learner Role", data.get("learner_role")),
            block("Voice Gender", data.get("voice_gender")),
            block("Voice Style", data.get("voice_style")),
            block("Introduction Voice Gender", data.get("intro_voice_gender")),
            block("Introduction Voice Style", data.get("intro_voice_style")),
            block("Introduction", data.get("introduction")),
            block("Opening Line", data.get("opening_line")),
            block("Beginning", data.get("beginning")),
            block("Beginning to Middle Cues", data.get("begin_to_middle_cues")),
            block("Middle", data.get("middle")),
            block("Middle to Ending Cues", data.get("middle_to_ending_cues")),
            block("Ending", data.get("ending")),
            block("Closing", data.get("closing")),
            block("Meta Instructions", data.get("meta_instructions")),
        ]
        return "\n".join(parts).strip() + "\n"

    def clean_voice_gender(self):
        value = (self.cleaned_data.get("voice_gender") or "").strip().lower()
        if value not in {"female", "male"}:
            return "female"
        return value

    def clean_intro_voice_gender(self):
        value = (self.cleaned_data.get("intro_voice_gender") or "").strip().lower()
        if value in {"female", "male"}:
            return value
        base = (self.cleaned_data.get("voice_gender") or "female").strip().lower()
        return base if base in {"female", "male"} else "female"


class StudentAccountCreateForm(forms.Form):
    first_name = forms.CharField(max_length=150)
    last_name = forms.CharField(max_length=150)
    net_id = forms.CharField(max_length=150, help_text="Required. Used for both username and initial password.")
    student_id = forms.CharField(max_length=30, help_text="Numbers only")
    class_name = forms.CharField(
        max_length=150,
        required=False,
        help_text='Optional. Leave blank to assign "Unassigned".',
    )

    def clean_student_id(self):
        value = self.cleaned_data["student_id"].strip()
        if not value.isdigit():
            raise forms.ValidationError("Student ID must be numeric.")
        return value

    def clean_net_id(self):
        value = self.cleaned_data["net_id"].strip()
        if not value:
            raise forms.ValidationError("NetID is required.")
        return value.lower()


class StudentBulkUploadForm(forms.Form):
    file = forms.FileField(
        help_text="Upload .xlsx or .csv with columns: first_name, last_name, net_id, student_id, class_name",
        widget=forms.ClearableFileInput(
            attrs={
                "accept": ".xlsx,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv"
            }
        ),
    )

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        suffix = uploaded.name.lower()
        if not (suffix.endswith(".xlsx") or suffix.endswith(".csv")):
            raise forms.ValidationError("Only .xlsx and .csv files are supported.")
        return uploaded


class ClassGroupCreateForm(forms.Form):
    class_name = forms.CharField(max_length=150)


class PromptTextUploadForm(forms.Form):
    title = forms.CharField(max_length=200, required=False, help_text="Optional. Defaults to file name.")
    file = forms.FileField(
        help_text="Upload .txt or .md",
        widget=forms.ClearableFileInput(attrs={"accept": ".txt,.md,text/plain,text/markdown"}),
    )
    is_active = forms.BooleanField(required=False)

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        suffix = uploaded.name.lower()
        if not (suffix.endswith(".txt") or suffix.endswith(".md")):
            raise forms.ValidationError("Only .txt or .md files are supported.")
        return uploaded
