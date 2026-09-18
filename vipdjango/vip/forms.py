from django import forms
from .models import RolePrompt
from .prompt_utils import find_section_by_aliases, normalize_gender, split_markdown_sections


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
        required=False,
        widget=forms.Textarea(attrs={"rows": 8}),
        help_text="Scenario facts, circumstances, beliefs, feelings, and experiences.",
    )
    learner_role = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Who is the human participant in this simulation.",
    )
    introduction = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 5}),
        help_text="Optional fixed message shown when a new chat starts.",
    )
    conversation_objectives = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 10}),
        help_text=(
            "Use one ### heading per concern, with possible expressions and resolution guidance underneath."
        ),
    )
    voice_gender = forms.ChoiceField(choices=[("female", "female"), ("male", "male")], initial="female")
    voice_style = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="Optional speaking style. Example: warm, calm, relatively slow.",
    )
    opening_line = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. Shown as the first in-character response after the learner greets the character.",
    )
    beginning = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 6}))
    begin_to_middle_cues = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. One cue per line (bullet points are fine).",
    )
    middle = forms.CharField(
        required=False, widget=forms.Textarea(attrs={"rows": 8}),
        help_text="For guided scenario practice, use a top-level '- Theme' bullet for each theme and indented numbered questions beneath it. Use the full question set and conditional reactions. The app tracks concerns, answers learner questions, and allows up to 25 exchanges with time reserved for each theme.",
    )
    middle_to_ending_cues = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="Optional. One cue per line (bullet points are fine).",
    )
    ending = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 6}))
    closing = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="Optional scenario-provided closing guidance.",
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
        sections = split_markdown_sections(content)
        voice_gender = normalize_gender(find_section_by_aliases(sections, ["voice gender"]), default="female")
        role = find_section_by_aliases(sections, ["role", "role summary", "character"])
        if not role:
            role = (content or "").strip()
        background = find_section_by_aliases(sections, ["background and context", "background", "context"])
        introduction = find_section_by_aliases(sections, ["introduction", "introduction: greeting"])
        legacy_meta = find_section_by_aliases(
            sections,
            ["meta instructions", "meta-instructions", "meta instruction", "notes"],
        )
        if legacy_meta:
            background = "\n\n".join(value for value in (background, "Scenario constraints:\n" + legacy_meta) if value)
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
            "learner_role": find_section_by_aliases(sections, ["learner role", "user role"]),
            "introduction": introduction,
            "conversation_objectives": find_section_by_aliases(
                sections, ["conversation objectives", "conversation goals", "objectives", "goals"]
            ),
            "voice_gender": voice_gender,
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
            block("Voice Gender", data.get("voice_gender")),
            block("Voice Style", data.get("voice_style")),
        ]
        return "\n".join(parts).strip() + "\n"

    def clean_voice_gender(self):
        value = (self.cleaned_data.get("voice_gender") or "").strip().lower()
        if value not in {"female", "male"}:
            return "female"
        return value


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
