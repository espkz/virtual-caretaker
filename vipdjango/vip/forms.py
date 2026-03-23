from django import forms
from .models import RolePrompt


class RolePromptForm(forms.ModelForm):
    class Meta:
        model = RolePrompt
        fields = ["title", "content", "is_active"]


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
    class_name = forms.CharField(max_length=150, help_text='Example: "NURS-101 Section A"')


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
