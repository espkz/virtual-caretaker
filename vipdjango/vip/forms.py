from django import forms
from .models import RolePrompt

class RolePromptForm(forms.ModelForm):
    class Meta:
        model = RolePrompt
        fields = ["title", "content", "is_active"]