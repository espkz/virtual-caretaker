from django.conf import settings
from django.db import models


class RolePrompt(models.Model):
    title = models.CharField(max_length=120)
    content = models.TextField()
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_prompts",
    )
    is_active = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class ChatSession(models.Model):
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
    )
    role_prompt = models.ForeignKey(
        RolePrompt,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sessions",
    )
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Session {self.id} - {self.student}"


class ChatMessage(models.Model):
    class Sender(models.TextChoices):
        STUDENT = "student", "Student"
        ASSISTANT = "assistant", "Assistant"

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sender = models.CharField(max_length=20, choices=Sender.choices)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender} @ {self.created_at:%Y-%m-%d %H:%M}"
