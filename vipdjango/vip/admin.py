from django.contrib import admin

from .models import ChatMessage, ChatSession, RolePrompt


@admin.register(RolePrompt)
class RolePromptAdmin(admin.ModelAdmin):
    list_display = ("title", "is_active", "created_by", "updated_at")
    list_filter = ("is_active", "updated_at")
    search_fields = ("title", "content", "created_by__username")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "role_prompt", "started_at", "ended_at")
    list_filter = ("started_at", "ended_at", "role_prompt")
    search_fields = ("student__username", "role_prompt__title")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "sender", "created_at")
    list_filter = ("sender", "created_at")
    search_fields = ("content", "session__student__username")
