import uuid

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
    def missing_required_fields(self):
        from .prompt_utils import missing_required_prompt_fields
        return missing_required_prompt_fields(self.title, self.content)

    @property
    def is_complete(self):
        return not self.missing_required_fields

    def save(self, *args, **kwargs):
        """Incomplete prompts are always drafts and cannot be activated."""
        if self.is_active and not self.is_complete:
            self.is_active = False
            update_fields = kwargs.get("update_fields")
            if update_fields is not None:
                kwargs["update_fields"] = set(update_fields) | {"is_active"}
        return super().save(*args, **kwargs)

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
        return "Core-question practice" if self.uses_core_questions else "Open-ended legacy scenario"


class ChatSession(models.Model):
    class InteractionMode(models.TextChoices):
        TEXT = "text", "Text"
        VOICE = "voice", "Voice"

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
    interaction_mode = models.CharField(
        max_length=10,
        choices=InteractionMode.choices,
        default=InteractionMode.TEXT,
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


class VoiceConversation(models.Model):
    """Maps one ElevenLabs voice conversation to the durable Django chat."""

    class Status(models.TextChoices):
        TOKEN_ISSUED = "token_issued", "Token issued"
        CONNECTED = "connected", "Connected"
        ENDED = "ended", "Ended"
        ERROR = "error", "Error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chat_session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="voice_conversations",
    )
    # The provider gives us this ID when the token is minted in normal flows.
    # It remains blank briefly when an older SDK/API only supplies the ID after
    # the browser connection is established; the bind endpoint fills it then.
    provider_conversation_id = models.CharField(max_length=128, blank=True, default="")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.TOKEN_ISSUED,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    connected_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    failure_reason = models.CharField(max_length=240, blank=True, default="")
    is_muted = models.BooleanField(default=False)
    # This is an explicit, durable handoff from the browser narrator to live
    # Speech Engine input.  Until it is set, finalized provider transcripts
    # are intentionally ignored by the adapter.
    introduction_completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["provider_conversation_id"],
                condition=~models.Q(provider_conversation_id=""),
                name="unique_nonempty_voice_provider_conversation",
            ),
        ]

    def __str__(self):
        return f"Voice {self.id} for chat {self.chat_session_id}"


class SpeechEngineVoiceResource(models.Model):
    """Persistent voice-to-resource mapping for Speech Engine's fixed TTS voice."""

    voice_id = models.CharField(max_length=128, unique=True)
    speech_engine_id = models.CharField(max_length=128, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.voice_id} -> {self.speech_engine_id}"
