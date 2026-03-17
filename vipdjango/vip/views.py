import os
import re
import io
import csv
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.db.models import Count, Max
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import ClassGroupCreateForm, RolePromptForm, StudentAccountCreateForm, StudentBulkUploadForm
from .models import ChatMessage, ChatSession, RolePrompt

"""
base
"""


def _build_chat_prompt(role_text, session_messages):
    prompt_template_path = Path(settings.BASE_DIR).parent / "prompts" / "prompt_template.md"
    if prompt_template_path.exists():
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
    else:
        prompt_template = "{role}\n\nConversation:\n"

    conversation_lines = []
    for message in session_messages:
        label = "Student" if message.sender == ChatMessage.Sender.STUDENT else "Assistant"
        conversation_lines.append(f"{label}: {message.content}")

    return prompt_template.format(role=role_text) + "\n" + "\n".join(conversation_lines)


def _load_api_key_from_txt():
    api_key_path = Path(settings.BASE_DIR).parent / "api_key.txt"
    if not api_key_path.exists():
        return ""
    return api_key_path.read_text(encoding="utf-8").strip()


def _parse_dialogue_and_emotion(text):
    match = re.match(r"^(.*?)(\[.*\])?$", text.strip(), re.DOTALL)
    if not match:
        return text, ""
    dialogue = match.group(1).strip()
    emotion = match.group(2).strip("[]") if match.group(2) else ""
    return dialogue, emotion


def _clean_for_tts(text):
    return re.sub(r"\([^)]*\)", "", text).strip()


def _generate_assistant_response(role_text, session_messages):
    api_key = _load_api_key_from_txt()
    if not api_key:
        return "OpenAI API key is not configured on the server yet."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = _build_chat_prompt(role_text, session_messages)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
        )
        return response.output_text or "The assistant returned an empty response."
    except Exception as exc:
        return f"Assistant error: {exc}"


def _clean_name_part(value):
    return re.sub(r"[^a-zA-Z0-9]", "", value or "").lower()


def _build_student_username(first_name, last_name):
    first = _clean_name_part(first_name)
    last = _clean_name_part(last_name)
    if not first and not last:
        return "student"
    if first and last:
        return f"{first}.{last}"
    return first or last


def _build_student_password(first_name, last_name, student_id):
    return f"{_clean_name_part(first_name)}{_clean_name_part(last_name)}{student_id}"


def _normalize_class_name(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def _normalize_header(header):
    return re.sub(r"[^a-z0-9]", "", (header or "").strip().lower())


def _extract_row_value(row, aliases):
    normalized = {_normalize_header(k): v for k, v in row.items()}
    for alias in aliases:
        value = normalized.get(alias)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _read_bulk_rows(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        decoded = uploaded_file.read().decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(decoded)))

    if suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:
            raise ValueError("openpyxl is required to read .xlsx files.") from exc

        wb = openpyxl.load_workbook(uploaded_file, data_only=True)
        ws = wb.active
        headers = [str(cell.value).strip() if cell.value is not None else "" for cell in ws[1]]
        rows = []
        for values in ws.iter_rows(min_row=2, values_only=True):
            if all(v is None or str(v).strip() == "" for v in values):
                continue
            rows.append(
                {
                    headers[i]: ("" if values[i] is None else str(values[i]).strip())
                    for i in range(len(headers))
                }
            )
        return rows

    raise ValueError("Unsupported file type. Please upload .xlsx or .csv.")


def _create_student_account(first_name, last_name, student_id, class_name):
    User = get_user_model()
    username = _build_student_username(first_name, last_name)
    password = _build_student_password(first_name, last_name, student_id)
    normalized_class = _normalize_class_name(class_name)

    if User.objects.filter(username=username).exists():
        return None, f'Username "{username}" already exists.'

    user = User.objects.create_user(
        username=username,
        password=password,
        first_name=first_name.strip(),
        last_name=last_name.strip(),
    )

    student_group, _ = Group.objects.get_or_create(name="Student")
    class_group, _ = Group.objects.get_or_create(name=f"Class: {normalized_class}")
    user.groups.add(student_group, class_group)
    return {
        "username": username,
        "password": password,
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "student_id": student_id.strip(),
        "class_name": normalized_class,
    }, None


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
        .prefetch_related("groups")
        .annotate(
            session_count=Count("chat_sessions", distinct=True),
            last_session_at=Max("chat_sessions__started_at"),
        )
        .order_by("username")
    )
    students_by_class = {}
    for student in students:
        class_names = []
        for group in student.groups.all():
            if group.name.startswith("Class: "):
                class_names.append(group.name.replace("Class: ", "", 1))
        if not class_names:
            class_names = ["Unassigned"]
        for class_name in class_names:
            students_by_class.setdefault(class_name, []).append(student)
    students_by_class = [
        {"class_name": class_name, "students": students_by_class[class_name]}
        for class_name in sorted(students_by_class.keys())
    ]
    return render(
        request,
        "vip/professor_dashboard.html",
        {
            "prompts": prompts,
            "students_by_class": students_by_class,
            "current_tab": current_tab,
        },
    )


@login_required
def professor_manage_accounts(request):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    single_form = StudentAccountCreateForm()
    bulk_form = StudentBulkUploadForm()
    class_form = ClassGroupCreateForm()
    created_accounts = []
    skipped_rows = []
    info_message = ""

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "create_single":
            single_form = StudentAccountCreateForm(request.POST)
            if single_form.is_valid():
                account, error = _create_student_account(
                    first_name=single_form.cleaned_data["first_name"],
                    last_name=single_form.cleaned_data["last_name"],
                    student_id=single_form.cleaned_data["student_id"],
                    class_name=single_form.cleaned_data["class_name"],
                )
                if error:
                    skipped_rows.append({"row": "Single form", "reason": error})
                else:
                    created_accounts.append(account)
                    info_message = "Student account created."

        if action == "bulk_upload":
            bulk_form = StudentBulkUploadForm(request.POST, request.FILES)
            if bulk_form.is_valid():
                aliases = {
                    "first_name": ["firstname", "first", "givenname"],
                    "last_name": ["lastname", "last", "surname", "familyname"],
                    "student_id": ["studentid", "id", "studentnumber", "sid"],
                    "class_name": ["classname", "class", "course", "section", "group"],
                }
                try:
                    rows = _read_bulk_rows(bulk_form.cleaned_data["file"])
                except ValueError as exc:
                    skipped_rows.append({"row": "Upload", "reason": str(exc)})
                    rows = []

                for idx, row in enumerate(rows, start=2):
                    first_name = _extract_row_value(row, aliases["first_name"])
                    last_name = _extract_row_value(row, aliases["last_name"])
                    student_id = _extract_row_value(row, aliases["student_id"])
                    class_name = _extract_row_value(row, aliases["class_name"])
                    if re.match(r"^\d+\.0$", student_id):
                        student_id = student_id[:-2]

                    if not first_name or not last_name or not student_id or not class_name:
                        skipped_rows.append(
                            {"row": idx, "reason": "Missing required fields (first_name, last_name, student_id, class_name)."}
                        )
                        continue
                    if not student_id.isdigit():
                        skipped_rows.append({"row": idx, "reason": "student_id must be numeric."})
                        continue

                    account, error = _create_student_account(
                        first_name=first_name,
                        last_name=last_name,
                        student_id=student_id,
                        class_name=class_name,
                    )
                    if error:
                        skipped_rows.append({"row": idx, "reason": error})
                    else:
                        created_accounts.append(account)

                info_message = f"Bulk upload complete. Created {len(created_accounts)} account(s)."

        if action == "create_class":
            class_form = ClassGroupCreateForm(request.POST)
            if class_form.is_valid():
                normalized_class = _normalize_class_name(class_form.cleaned_data["class_name"])
                Group.objects.get_or_create(name=f"Class: {normalized_class}")
                info_message = f'Class group "{normalized_class}" is ready.'

        if action == "move_student":
            student_id = request.POST.get("student_id", "").strip()
            class_name = _normalize_class_name(request.POST.get("class_name", ""))
            User = get_user_model()
            student = get_object_or_404(User, pk=student_id, groups__name="Student")
            if class_name:
                class_group, _ = Group.objects.get_or_create(name=f"Class: {class_name}")
                existing_class_groups = student.groups.filter(name__startswith="Class: ")
                student.groups.remove(*existing_class_groups)
                student.groups.add(class_group)
                info_message = f"Moved {student.username} to class {class_name}."
            else:
                skipped_rows.append({"row": "Move student", "reason": "Class name cannot be empty."})

        if action == "delete_student":
            student_id = request.POST.get("student_id", "").strip()
            User = get_user_model()
            student = get_object_or_404(User, pk=student_id, groups__name="Student")
            username = student.username
            student.delete()
            info_message = f"Deleted student account {username}."

    class_groups = Group.objects.filter(name__startswith="Class: ").order_by("name")
    User = get_user_model()
    roster_students = User.objects.filter(groups__name="Student").prefetch_related("groups").order_by("username")
    student_rows = []
    for student in roster_students:
        class_names = [
            group.name.replace("Class: ", "", 1)
            for group in student.groups.all()
            if group.name.startswith("Class: ")
        ]
        student_rows.append(
            {
                "id": student.id,
                "username": student.username,
                "first_name": student.first_name,
                "last_name": student.last_name,
                "classes_display": ", ".join(class_names) if class_names else "Unassigned",
            }
        )

    return render(
        request,
        "vip/professor_accounts.html",
        {
            "single_form": single_form,
            "bulk_form": bulk_form,
            "class_form": class_form,
            "created_accounts": created_accounts,
            "skipped_rows": skipped_rows,
            "info_message": info_message,
            "class_groups": class_groups,
            "student_rows": student_rows,
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


@login_required
@require_POST
def professor_delete_session(request, session_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    session = get_object_or_404(ChatSession, pk=session_id)
    student_id = session.student_id
    session.delete()

    next_url = request.POST.get("next")
    if next_url:
        return redirect(next_url)
    return redirect("vip:professor_student_logs", student_id=student_id)


@login_required
@require_POST
def professor_delete_student_logs(request, student_id):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    User = get_user_model()
    student = get_object_or_404(User, pk=student_id, groups__name="Student")
    ChatSession.objects.filter(student=student).delete()
    return redirect("vip:professor_student_logs", student_id=student_id)


@login_required
@require_POST
def professor_reset_all_student_logs(request):
    if not request.user.groups.filter(name="Professor").exists():
        return redirect("vip:home")

    User = get_user_model()
    student_ids = User.objects.filter(groups__name="Student").values_list("id", flat=True)
    ChatSession.objects.filter(student_id__in=student_ids).delete()
    return redirect(f"{reverse('vip:professor_dashboard')}?tab=logs")


"""
STUDENT
"""

@login_required
def student_dashboard(request):
    if not request.user.groups.filter(name="Student").exists():
        return redirect("vip:home")

    active_prompts = RolePrompt.objects.filter(is_active=True).order_by("title")
    sessions = (
        ChatSession.objects.filter(student=request.user)
        .select_related("role_prompt")
        .prefetch_related("messages")
        .order_by("-started_at")
    )

    selected_prompt = None
    selected_prompt_id = request.POST.get("prompt_id") or request.GET.get("prompt")
    if selected_prompt_id:
        selected_prompt = active_prompts.filter(pk=selected_prompt_id).first()
    if not selected_prompt:
        selected_prompt = active_prompts.first()

    start_new = request.GET.get("new") == "1"
    force_new = request.POST.get("force_new") == "1" or request.GET.get("force_new") == "1"
    current_session = None
    session_id = request.POST.get("session_id") or request.GET.get("session")
    if session_id:
        current_session = get_object_or_404(
            ChatSession.objects.filter(student=request.user).select_related("role_prompt"),
            pk=session_id,
        )
    elif not start_new and selected_prompt:
        current_session = sessions.filter(role_prompt=selected_prompt).first()

    error_message = None
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "new_session":
            target = reverse("vip:student_dashboard")
            if selected_prompt:
                target += f"?prompt={selected_prompt.id}&new=1&force_new=1"
            return redirect(target)

        if action == "close_conversation":
            if current_session and current_session.ended_at is None:
                current_session.ended_at = timezone.now()
                current_session.save(update_fields=["ended_at"])
            target = reverse("vip:student_dashboard")
            if selected_prompt:
                target += f"?prompt={selected_prompt.id}&new=1"
            return redirect(target)

        if action == "send_message":
            user_text = request.POST.get("message", "").strip()
            if not selected_prompt:
                error_message = "No active prompt is available. Ask your professor to activate one."
            elif not user_text:
                error_message = "Please type a message before sending."
            else:
                if (
                    not current_session
                    or current_session.role_prompt_id != selected_prompt.id
                    or current_session.ended_at is not None
                    or force_new
                ):
                    current_session = None
                    if not force_new:
                        current_session = (
                            ChatSession.objects.filter(
                                student=request.user,
                                role_prompt=selected_prompt,
                                ended_at__isnull=True,
                            )
                            .order_by("-started_at")
                            .first()
                        )
                    if current_session is None:
                        current_session = ChatSession.objects.create(
                            student=request.user,
                            role_prompt=selected_prompt,
                        )
                else:
                    latest_message = current_session.messages.order_by("-created_at").first()
                    if latest_message and latest_message.sender == ChatMessage.Sender.STUDENT:
                        error_message = "Please wait for the AI response before sending another message."

                if error_message:
                    current_messages = []
                    if current_session:
                        current_messages = current_session.messages.order_by("created_at")
                    rendered_messages = []
                    for message in current_messages:
                        display_content = message.content
                        if message.sender == ChatMessage.Sender.ASSISTANT:
                            display_content, _ = _parse_dialogue_and_emotion(message.content)
                        rendered_messages.append(
                            {
                                "id": message.id,
                                "sender": message.sender,
                                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else "Student",
                                "created_at": message.created_at,
                                "display_content": display_content,
                            }
                        )
                    return render(
                        request,
                        "vip/student_dashboard.html",
                        {
                            "active_prompts": active_prompts,
                            "selected_prompt": selected_prompt,
                            "sessions": sessions,
                            "current_session": current_session,
                            "rendered_messages": rendered_messages,
                            "error_message": error_message,
                        },
                    )

                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.STUDENT,
                    content=user_text,
                )

                session_messages = current_session.messages.order_by("created_at")
                assistant_text = _generate_assistant_response(selected_prompt.content, session_messages)
                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.ASSISTANT,
                    content=assistant_text,
                )
                return redirect(
                    f"{reverse('vip:student_dashboard')}?prompt={selected_prompt.id}&session={current_session.id}&autoplay=1"
                )

    current_messages = []
    if current_session:
        current_messages = current_session.messages.order_by("created_at")
    rendered_messages = []
    for message in current_messages:
        display_content = message.content
        if message.sender == ChatMessage.Sender.ASSISTANT:
            display_content, _ = _parse_dialogue_and_emotion(message.content)
        rendered_messages.append(
            {
                "id": message.id,
                "sender": message.sender,
                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else "Student",
                "created_at": message.created_at,
                "display_content": display_content,
            }
        )

    return render(
        request,
        "vip/student_dashboard.html",
        {
            "active_prompts": active_prompts,
            "selected_prompt": selected_prompt,
            "sessions": sessions,
            "current_session": current_session,
            "rendered_messages": rendered_messages,
            "error_message": error_message,
            "force_new": force_new,
        },
    )


@login_required
def student_download_session(request, session_id):
    if not request.user.groups.filter(name="Student").exists():
        return redirect("vip:home")

    session = get_object_or_404(
        ChatSession.objects.select_related("student", "role_prompt"),
        pk=session_id,
        student=request.user,
    )
    messages = session.messages.order_by("created_at")

    lines = [
        f"Student: {session.student.username}",
        f"Prompt: {session.role_prompt.title if session.role_prompt else 'None'}",
        f"Session ID: {session.id}",
        f"Started: {session.started_at}",
        "",
    ]
    for message in messages:
        speaker = "Student" if message.sender == ChatMessage.Sender.STUDENT else "Assistant"
        lines.append(f"[{message.created_at}] {speaker}: {message.content}")
        lines.append("")

    response = HttpResponse("\n".join(lines), content_type="text/plain")
    response["Content-Disposition"] = f'attachment; filename="chat_session_{session.id}.txt"'
    return response


@login_required
def student_message_tts(request, message_id):
    if not request.user.groups.filter(name="Student").exists():
        return redirect("vip:home")

    message = get_object_or_404(
        ChatMessage.objects.select_related("session"),
        pk=message_id,
        session__student=request.user,
    )
    if message.sender != ChatMessage.Sender.ASSISTANT:
        return HttpResponseBadRequest("TTS is only available for assistant messages.")

    api_key = _load_api_key_from_txt()
    if not api_key:
        return HttpResponseBadRequest("API key is not configured.")

    dialogue, emotion = _parse_dialogue_and_emotion(message.content)
    use_emotion_voice = request.GET.get("emotion", "1") != "0"

    try:
        if use_emotion_voice:
            from openai import OpenAI

            voice = "coral"
            if re.search(r"\bmale voice\b", emotion.lower()):
                voice = "onyx"

            tts_client = OpenAI(api_key=api_key)
            response = tts_client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=dialogue or message.content,
                instructions=emotion or "Speak naturally and clearly.",
            )
            return HttpResponse(response.read(), content_type="audio/mpeg")

        from gtts import gTTS

        clean_text = _clean_for_tts(dialogue or message.content)
        audio_buffer = io.BytesIO()
        tts = gTTS(text=clean_text or "No content", lang="en")
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return HttpResponse(audio_buffer.read(), content_type="audio/mpeg")
    except Exception as exc:
        return HttpResponseBadRequest(f"TTS error: {exc}")
