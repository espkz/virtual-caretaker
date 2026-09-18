from django import forms
from .models import RolePrompt
from .prompt_utils import (
    find_exact_section_by_aliases,
    find_section_by_aliases,
    normalize_voice_style,
    split_markdown_sections,
)


class RolePromptForm(forms.Form):
    title = forms.CharField(max_length=120)
    is_active = forms.BooleanField(required=False)
    simulation_mode = forms.ChoiceField(
        choices=[("roleplay", "Patient or family roleplay"), ("clinician_demo", "Clinician demonstration")],
        initial="roleplay", required=False,
        help_text="Clinician demonstration answers the human's concerns. Also set the Role, User Role, guidance, and closing for that clinician; this choice does not rewrite them.",
    )

    role = forms.CharField(widget=forms.Textarea(attrs={"rows": 5}))
    background_context = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 8}),
        help_text="Scenario facts, circumstances, beliefs, feelings, and experiences.",
    )
    meta_instructions = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 10}),
        help_text=(
            "Optional author guidance for the character's behavior and response boundaries. "
            "This is reference guidance, not spoken dialogue."
        ),
    )
    learner_role = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Who is the human participant in this simulation.",
    )
    introduction = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 5}),
        help_text="Fixed message shown when a new chat starts.",
    )
    conversation_objectives = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 10}),
        help_text=(
            "Use one ### heading per concern, with possible expressions and resolution guidance underneath."
        ),
    )
    introduction_voice_id = forms.ChoiceField(
        choices=(),
        help_text="Voice used only for the simulation introduction.",
    )
    roleplay_voice_id = forms.ChoiceField(
        choices=(),
        help_text="Voice used for the roleplay character during the conversation.",
    )
    voice_style = forms.CharField(
        max_length=240,
        widget=forms.TextInput(),
        help_text="Add one- or two-word delivery tags. Tags are stored in Markdown separated by commas.",
    )
    opening_line = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Shown as the first in-character response after the learner greets the character.",
    )
    beginning = forms.CharField(widget=forms.Textarea(attrs={"rows": 6}))
    begin_to_middle_cues = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="One cue per line (bullet points are fine).",
    )
    middle = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 8}),
        help_text="For guided scenario practice, use a top-level '- Theme' bullet for each theme and indented numbered questions beneath it. Use the full question set and conditional reactions. The app tracks concerns, answers learner questions, and allows up to 25 exchanges with time reserved for each theme.",
    )
    middle = forms.CharField(
        required=False, widget=forms.Textarea(attrs={"rows": 8}),
        help_text="For guided scenario practice, use a top-level '- Theme' bullet for each theme and indented numbered questions beneath it. Use the full question set and conditional reactions. The app tracks concerns, answers learner questions, and allows up to 25 exchanges with time reserved for each theme.",
    )
    middle_to_ending_cues = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="One cue per line (bullet points are fine).",
    )
    ending = forms.CharField(widget=forms.Textarea(attrs={"rows": 6}))
    closing = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="Scenario-provided closing guidance.",
    )

    def __init__(self, *args, voice_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        choices = [("", "Select an ElevenLabs voice")]
        choices.extend(list(voice_choices or ()))
        known_ids = {value for value, _label in choices}
        for field_name in ("introduction_voice_id", "roleplay_voice_id"):
            selected = ""
            if self.is_bound:
                selected = (self.data.get(field_name) or "").strip()
            elif self.initial:
                selected = (self.initial.get(field_name) or "").strip()
            field_choices = list(choices)
            if selected and selected not in known_ids:
                field_choices.append((selected, "Saved voice (provider list currently unavailable)"))
            self.fields[field_name].choices = field_choices

    @classmethod
    def initial_from_prompt(cls, prompt):
        return cls.initial_from_content(
            prompt.content,
            title=prompt.title,
            is_active=prompt.is_active,
        )

    @classmethod
    def initial_from_content(cls, content, title="", is_active=False):
        sections = split_markdown_sections(content)
        role = find_section_by_aliases(sections, ["role", "role summary", "character"])
        if not role:
            role = (content or "").strip()
        background = find_section_by_aliases(sections, ["background and context", "background", "context"])
        meta_instructions = find_section_by_aliases(
            sections,
            ["meta instructions", "meta-instructions", "meta instruction", "notes", "constraints", "rules"],
        )
        introduction = find_section_by_aliases(sections, ["introduction", "introduction: greeting"])
        ending = find_section_by_aliases(
            sections,
            ["ending", "end", "conversation progression: end", "conversation progression: ending"],
        )
        legacy_end_cues = find_section_by_aliases(sections, ["end of conversation cues"])
        if legacy_end_cues:
            ending = "\n\n".join(value for value in (ending, "End-of-conversation guidance:\n" + legacy_end_cues) if value)
        return {
            "title": title,
            "simulation_mode": find_section_by_aliases(sections, ["simulation mode"]).strip().lower() or "roleplay",
            "is_active": is_active,
            "role": role,
            "background_context": background,
            "meta_instructions": meta_instructions,
            "learner_role": find_section_by_aliases(sections, ["learner role", "user role"]),
            "introduction": introduction,
            "conversation_objectives": find_section_by_aliases(
                sections, ["conversation objectives", "conversation goals", "objectives", "goals"]
            ),
            "introduction_voice_id": find_exact_section_by_aliases(
                sections, ["introduction voice", "narrator voice"]
            ).strip(),
            "roleplay_voice_id": find_exact_section_by_aliases(
                sections, ["roleplay voice", "character voice"]
            ).strip(),
            "voice_style": find_section_by_aliases(sections, ["voice style", "voice instructions"]),
            "opening_line": find_section_by_aliases(sections, ["opening line"]),
            "beginning": find_section_by_aliases(sections, ["beginning", "conversation progression: beginning"]),
            "begin_to_middle_cues": find_section_by_aliases(
                sections,
                ["beginning to middle transition", "beginning to middle cues", "middle triggers", "middle trigger", "trigger"],
            ),
            "middle": find_section_by_aliases(sections, ["middle", "conversation progression: middle"]),
            "middle_to_ending_cues": find_section_by_aliases(
                sections,
                ["middle to ending transition", "middle to ending cues", "ending triggers", "ending trigger"],
            ),
            "ending": ending,
            "closing": find_section_by_aliases(sections, ["closing", "final response"]),
        }

    def render_markdown_content(self):
        data = self.cleaned_data

        def block(header, value):
            return f"## {header}\n{(value or '').strip()}\n"

        def subsection(header, value):
            return f"### {header}\n{(value or '').strip()}\n"

        parts = [
            block("Simulation Mode", data.get("simulation_mode") or "roleplay"),
            block("Role", data.get("role")),
            block("Background and Context", data.get("background_context")),
            block("Meta Instructions", data.get("meta_instructions")),
            block("User Role", data.get("learner_role")),
            block("Conversation Goals", data.get("conversation_objectives")),
            block("Introduction", data.get("introduction")),
            "## Conversation Stages\n",
            subsection("Opening Line", data.get("opening_line")),
            subsection("Beginning", data.get("beginning")),
            subsection("Beginning to Middle Transition", data.get("begin_to_middle_cues")),
            subsection("Middle", data.get("middle")),
            subsection("Middle to Ending Transition", data.get("middle_to_ending_cues")),
            subsection("Ending", data.get("ending")),
            block("Closing", data.get("closing")),
            block("Introduction Voice", data.get("introduction_voice_id")),
            block("Roleplay Voice", data.get("roleplay_voice_id")),
            block("Voice Style", data.get("voice_style")),
        ]
        return "\n".join(parts).strip() + "\n"

    def clean_voice_style(self):
        value = (self.cleaned_data.get("voice_style") or "").strip()
        normalized = normalize_voice_style(value, default="")
        if not normalized:
            raise forms.ValidationError(
                "Each voice-style tag must contain one or two words and cannot contain role, state, or system instructions."
            )
        return normalized


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
