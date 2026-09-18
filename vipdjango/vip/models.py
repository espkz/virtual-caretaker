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

    @property
    def uses_core_questions(self):
        from .conversation_scenario import parse_scenario_prompt
        from .core_questions import enabled
        return enabled(parse_scenario_prompt(self.content))

    @property
    def simulation_mode_label(self):
        from .conversation_scenario import parse_scenario_prompt
        if parse_scenario_prompt(self.content).simulation_mode == "clinician_demo":
            return "Clinician demonstration"
        return "Guided scenario practice" if self.uses_core_questions else "Open-ended legacy scenario"


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
    conversation_stage = models.CharField(max_length=20, default="beginning")
    conversation_phase = models.CharField(max_length=32, default="normal")
    scenario_content = models.TextField(blank=True, default="")
    core_question_state = models.JSONField(default=dict, blank=True)
    completion_status = models.BooleanField(default=False)
    active_objective = models.CharField(max_length=160, blank=True, default="")
    covered_objectives = models.JSONField(default=list, blank=True)
    unresolved_objectives = models.JSONField(default=list, blank=True)
    recent_topics = models.JSONField(default=list, blank=True)
    # Concrete concern clusters parsed from the scenario's Middle guidance.
    # These are monotonic application state; the model reports context, while
    # the application advances the ledger and prevents reopening a topic.
    active_topic = models.CharField(max_length=160, blank=True, default="")
    covered_topics = models.JSONField(default=list, blank=True)
    unresolved_topics = models.JSONField(default=list, blank=True)
    topic_turn_counts = models.JSONField(default=dict, blank=True)
    ending_ready = models.BooleanField(default=False)
    # Only one learner turn may be in flight for a session.  The claim is
    # cleared when the matching assistant response is committed or the
    # request is released after an error/cancellation.
    active_turn_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    active_claim_id = models.CharField(max_length=64, blank=True, default="")
    active_claimed_at = models.DateTimeField(null=True, blank=True)
    last_completed_turn_id = models.CharField(max_length=64, blank=True, default="")

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
    voice_metadata = models.TextField(blank=True, default="")
    # A client-generated ID ties the learner message to exactly one assistant
    # response. Blank keeps legacy rows valid; new turns always provide it.
    turn_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["session", "sender", "turn_id"],
                condition=~models.Q(turn_id=""),
                name="unique_nonempty_message_turn_per_sender",
            ),
        ]

    def __str__(self):
        return f"{self.sender} @ {self.created_at:%Y-%m-%d %H:%M}"
