from django import forms
from .models import RolePrompt


class RolePromptForm(forms.ModelForm):
    class Meta:
        model = RolePrompt
        fields = ["title", "content", "is_active"]


class StudentAccountCreateForm(forms.Form):
    first_name = forms.CharField(max_length=150)
    last_name = forms.CharField(max_length=150)
    student_id = forms.CharField(max_length=30, help_text="Numbers only")
    class_name = forms.CharField(max_length=150, help_text='Example: "NURS-101 Section A"')

    def clean_student_id(self):
        value = self.cleaned_data["student_id"].strip()
        if not value.isdigit():
            raise forms.ValidationError("Student ID must be numeric.")
        return value


class StudentBulkUploadForm(forms.Form):
    file = forms.FileField(
        help_text="Upload .xlsx or .csv with columns: first_name, last_name, student_id, class_name"
    )


class ClassGroupCreateForm(forms.Form):
    class_name = forms.CharField(max_length=150, help_text='Example: "NURS-101 Section A"')
