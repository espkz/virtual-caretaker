from django.contrib import admin

from .models import ChatMessage, ChatSession, ProfessorClass, ProfessorStudent, RolePrompt


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


@admin.register(ProfessorClass)
class ProfessorClassAdmin(admin.ModelAdmin):
    list_display = ("name", "professor", "created_at")
    list_filter = ("created_at",)
    search_fields = ("name", "professor__username")


@admin.register(ProfessorStudent)
class ProfessorStudentAdmin(admin.ModelAdmin):
    list_display = ("student", "professor", "class_group", "student_number", "created_at")
    list_filter = ("class_group", "created_at")
    search_fields = ("student__username", "student__first_name", "student__last_name", "professor__username")
