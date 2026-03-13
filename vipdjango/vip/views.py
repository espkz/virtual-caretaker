from django.shortcuts import get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.db.models import Count, Max
from django.shortcuts import redirect
from django.views.decorators.http import require_POST
from .forms import RolePromptForm

from .models import ChatSession, RolePrompt

"""
base
"""

@login_required
def home(request):
    if request.user.groups.filter(name="Professor").exists():
        return redirect("vip:professor_dashboard")
    elif request.user.groups.filter(name="Student").exists():
        return redirect("vip:student_dashboard")
    else:
        return redirect("vip:dashboard")
    
@login_required
def dashboard(request):
    return render(request, "vip/dashboard.html")

"""
PROFESSOR
"""
@login_required
def professor_dashboard(request):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    current_tab = request.GET.get("tab", "prompts")
    if current_tab not in {"prompts", "logs"}:
        current_tab = "prompts"

    prompts = RolePrompt.objects.order_by("-updated_at")
    User = get_user_model()
    students = (
        User.objects.filter(groups__name="Student")
        .annotate(
            session_count=Count("chat_sessions", distinct=True),
            last_session_at=Max("chat_sessions__started_at"),
        )
        .order_by("username")
    )
    return render(
        request,
        "vip/professor_dashboard.html",
        {
            "prompts": prompts,
            "students": students,
            "current_tab": current_tab,
        },
    )

@login_required
def create_prompt(request):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    if request.method == "POST":
        form = RolePromptForm(request.POST)
        if form.is_valid():
            new_prompt = form.save(commit=False)
            new_prompt.created_by = request.user

            new_prompt.save()
            return redirect("vip:professor_dashboard")
    else:
        form = RolePromptForm()

    return render(
        request,
        "vip/prompt_form.html",
        {"form": form, "page_title": "Add Prompt"},
    )

@login_required
def edit_prompt(request, prompt_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    prompt = get_object_or_404(RolePrompt, pk=prompt_id)

    if request.method == "POST":
        form = RolePromptForm(request.POST, instance=prompt)
        if form.is_valid():
            form.save()
            return redirect("vip:professor_dashboard")
    else:
        form = RolePromptForm(instance=prompt)

    return render(
        request,
        "vip/prompt_form.html",
        {"form": form, "page_title": "Edit Prompt"},
    )


@login_required
@require_POST
def set_active_prompt(request, prompt_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    selected_prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    selected_prompt.is_active = True
    selected_prompt.save(update_fields=["is_active", "updated_at"])
    return redirect("vip:professor_dashboard")


@login_required
@require_POST
def deactivate_prompt(request, prompt_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    selected_prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    selected_prompt.is_active = False
    selected_prompt.save(update_fields=["is_active", "updated_at"])
    return redirect("vip:professor_dashboard")


@login_required
def professor_session_detail(request, session_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    session = get_object_or_404(
        ChatSession.objects.select_related("student", "role_prompt"),
        pk=session_id,
    )
    messages = session.messages.order_by("created_at")

    return render(
        request,
        "vip/professor_session_detail.html",
        {
            "session": session,
            "messages": messages,
        },
    )


@login_required
def professor_student_logs(request, student_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    User = get_user_model()
    student = get_object_or_404(User, pk=student_id, groups__name="Student")
    sessions = (
        ChatSession.objects.filter(student=student)
        .select_related("role_prompt")
        .prefetch_related("messages")
        .order_by("-started_at")
    )

    return render(
        request,
        "vip/professor_student_logs.html",
        {
            "student": student,
            "sessions": sessions,
        },
    )


"""
STUDENT
"""

@login_required
def student_dashboard(request):
    if not request.user.groups.filter(name="Student").exists():
        return redirect("vip:home")
    return render(request, "vip/student_dashboard.html")
