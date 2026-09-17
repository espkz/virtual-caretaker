from django.contrib import admin

from .models import ChatMessage, ChatSession, RolePrompt, SpeechEngineVoiceResource, VoiceConversation


@admin.register(RolePrompt)
class RolePromptAdmin(admin.ModelAdmin):
    list_display = ("title", "is_active", "created_by", "updated_at")
    list_filter = ("is_active", "updated_at")
    search_fields = ("title", "content", "created_by__username")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "role_prompt", "interaction_mode", "started_at", "ended_at")
    list_filter = ("interaction_mode", "started_at", "ended_at", "role_prompt")
    search_fields = ("student__username", "role_prompt__title")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "sender", "created_at")
    list_filter = ("sender", "created_at")
    search_fields = ("content", "session__student__username")


@admin.register(VoiceConversation)
class VoiceConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "chat_session", "provider_conversation_id", "status", "created_at", "ended_at")
    list_filter = ("status", "created_at", "ended_at")
    search_fields = ("provider_conversation_id", "chat_session__student__username")
    readonly_fields = ("id", "created_at", "connected_at", "ended_at")


@admin.register(SpeechEngineVoiceResource)
class SpeechEngineVoiceResourceAdmin(admin.ModelAdmin):
    list_display = ("voice_id", "speech_engine_id", "created_at")
    search_fields = ("voice_id", "speech_engine_id")
    readonly_fields = ("created_at",)
