import os
import re
import io
import csv
import json
import logging
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import Group
from django.contrib.auth import update_session_auth_hash
from django.db.models import Count, Max, Q
from django.urls import reverse
from django.utils.text import slugify
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import (
    ClassGroupCreateForm,
    PromptTextUploadForm,
    RolePromptForm,
    StudentAccountCreateForm,
    StudentBulkUploadForm,
)
from .models import ChatMessage, ChatSession, RolePrompt
from .prompt_utils import find_section_by_aliases, split_markdown_sections
from .conversation_graph import MAX_TURNS
from .conversation_engine import ConversationEngine, split_dialogue_and_voice
from .conversation_scenario import parse_scenario_prompt

logger = logging.getLogger(__name__)

def _prompt_file_path(filename):
    candidates = [
        Path(settings.BASE_DIR).parent / "prompts" / filename,
        Path(settings.BASE_DIR) / "prompts" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def _extract_template_prefix():
    prompt_template_path = _prompt_file_path("prompt_template.md")
    if prompt_template_path:
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
    else:
        prompt_template = ""
    if "{role}" in prompt_template:
        return prompt_template.split("{role}", 1)[0].strip()
    return prompt_template.strip()


def _parse_role_config(role_text):
    sections = split_markdown_sections(role_text)
    role = find_section_by_aliases(sections, ["role", "role summary", "character"])
    learner_role = find_section_by_aliases(sections, ["learner role", "user role"])
    voice_gender = (find_section_by_aliases(sections, ["voice gender", "voice"]) or "female").lower().strip()
    if voice_gender not in {"male", "female"}:
        merged = f"{role_text}\n{find_section_by_aliases(sections, ['introduction'])}"
        voice_gender = "male" if "male voice" in merged.lower() else "female"
    voice_style = find_section_by_aliases(sections, ["voice style", "voice instructions"]) or "speak naturally and clearly"
    intro_voice_gender = (
        find_section_by_aliases(sections, ["introduction voice gender", "intro voice gender"]) or voice_gender
    ).lower().strip()
    if intro_voice_gender not in {"male", "female"}:
        intro_voice_gender = voice_gender
    intro_voice_style = (
        find_section_by_aliases(sections, ["introduction voice style", "intro voice style"]) or voice_style
    )
    return {
        "role": role or role_text.strip(),
        "learner_role": learner_role or "nursing student",
        "voice_gender": voice_gender,
        "voice_style": voice_style.strip(),
        "intro_voice_gender": intro_voice_gender,
        "intro_voice_style": intro_voice_style.strip(),
        "introduction": find_section_by_aliases(sections, ["introduction", "introduction: greeting"]),
        "opening_line": find_section_by_aliases(sections, ["opening line"]),
        "beginning": find_section_by_aliases(sections, ["beginning", "conversation progression: beginning"]),
        "middle": find_section_by_aliases(sections, ["middle", "conversation progression: middle"]),
        "ending": find_section_by_aliases(sections, ["ending", "end", "conversation progression: end"]),
        "closing": find_section_by_aliases(sections, ["closing", "final response"]),
        "meta": find_section_by_aliases(sections, ["meta instructions", "meta instruction", "meta-instructions", "notes"]),
        "begin_cues": find_section_by_aliases(
            sections,
            ["beginning to middle cues", "begin-to-middle cues", "middle trigger", "middle triggers", "trigger"],
        ),
        "middle_to_ending_cues": find_section_by_aliases(
            sections, ["middle to ending cues", "middle-to-ending cues", "ending triggers", "ending trigger"]
        ),
        "end_of_conversation_cues": find_section_by_aliases(sections, ["end of conversation cues"]),
    }


def _enforce_voice_format(text, voice_gender, voice_style):
    text = (text or "").strip()
    if not text:
        return f"[{voice_gender} voice, {voice_style}]"
    if "[" not in text or "]" not in text:
        return f"{text}\n\n[{voice_gender} voice, {voice_style}]"
    found = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if not found:
        return f"{text}\n\n[{voice_gender} voice, {voice_style}]"
    raw = found[-1].strip()
    normalized = raw.lower().strip()
    bare_gender = {voice_gender, f"{voice_gender} voice", f"{voice_gender} voice, {voice_gender}"}
    if normalized in bare_gender:
        raw = voice_style
    elif not re.search(r"\b(?:male|female)\s+voice\b", normalized):
        raw = f"{voice_gender} voice, {raw}"
    return re.sub(
        r"\[([^\]]+)\]\s*$",
        f"[{raw}]",
        text,
        count=1,
        flags=re.DOTALL,
    )


def _conversation_messages(session_messages):
    """Convert persisted conversational messages into native LLM message roles."""
    messages = []
    for message in session_messages:
        if message.sender not in {ChatMessage.Sender.STUDENT, ChatMessage.Sender.ASSISTANT}:
            continue
        content = message.content
        if message.sender == ChatMessage.Sender.ASSISTANT:
            content, _ = split_dialogue_and_voice(content)
        messages.append({
            "role": "user" if message.sender == ChatMessage.Sender.STUDENT else "assistant",
            "content": content,
        })
    return messages


def _load_api_key_from_txt():
    # Preferred: environment variable for deployment safety.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    # Local file fallback (commented out for now).
    # api_key_path = Path(settings.BASE_DIR).parent / "api_key.txt"
    # if api_key_path.exists():
    #     return api_key_path.read_text(encoding="utf-8").strip()

    return ""


def _parse_dialogue_and_emotion(text):
    return split_dialogue_and_voice(text)


def _clean_for_tts(text):
    return re.sub(r"\([^)]*\)", "", text).strip()


def _detect_stop_request(message_text):
    """Use the model for semantic interrupt intent instead of phrase matching."""
    api_key = _load_api_key_from_txt()
    if not api_key or not message_text:
        return False
    try:
        from openai import OpenAI

        payload = [
            {
                "role": "system",
                "content": (
                    "Classify only whether the learner is clearly requesting that the conversation stop now. "
                    "Return true for an explicit request to end or stop the conversation. Return false when the "
                    "learner is continuing roleplay, discussing a scenario ending, expressing thanks while asking "
                    "another question, or otherwise has not clearly requested stopping."
                ),
            },
            {"role": "user", "content": message_text},
        ]
        logger.debug("LLM interrupt classifier payload:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5-nano",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "conversation_interrupt",
                    "schema": {
                        "type": "object",
                        "properties": {"stop_requested": {"type": "boolean"}},
                        "required": ["stop_requested"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
            input=payload,
        )
        return bool(json.loads(response.output_text or "{}").get("stop_requested"))
    except Exception:
        logger.exception("Unable to classify conversation interrupt request")
        return False


def _generate_assistant_response(role_text, session_messages, session=None):
    api_key = _load_api_key_from_txt()
    if not api_key:
        return "OpenAI API key is not configured on the server yet.", False, {"stage": "error", "reason": "missing_api_key"}
    engine = ConversationEngine(_extract_template_prefix(), api_key)
    conversation_state = None
    if session is not None:
        conversation_state = {
            "current_stage": session.conversation_stage,
            "conversation_stage": session.conversation_stage,
            "phase": session.conversation_phase,
            "completion_status": session.completion_status,
        }
    return engine.respond(role_text, list(session_messages), conversation_state=conversation_state)


def _clean_name_part(value):
    return re.sub(r"[^a-zA-Z0-9]", "", value or "").lower()


def _normalize_net_id(value):
    return (value or "").strip().lower()


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


def _create_student_account(first_name, last_name, net_id, student_id, class_name):
    User = get_user_model()
    username = _normalize_net_id(net_id)
    password = _build_student_password(first_name, last_name, student_id)
    normalized_class = _normalize_class_name(class_name)

    if not username:
        return None, "NetID is required."
    if User.objects.filter(username=username).exists():
        return None, f'NetID "{username}" already exists.'

    user = User.objects.create_user(
        username=username,
        password=password,
        first_name=first_name.strip(),
        last_name=last_name.strip(),
    )

    student_group, _ = Group.objects.get_or_create(name="Student")
    user.groups.add(student_group)
    if normalized_class:
        class_group, _ = Group.objects.get_or_create(name=f"Class: {normalized_class}")
        user.groups.add(class_group)
    return {
        "username": username,
        "password": password,
        "net_id": username,
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "student_id": student_id.strip(),
        "class_name": normalized_class or "Unassigned",
    }, None


def _is_professor(user):
    return user.is_superuser or user.groups.filter(name__iexact="Professor").exists()


def _is_student(user):
    if _is_professor(user):
        return False
    return (
        user.groups.filter(name__startswith="Class: ")
        .exclude(name__iexact="Class: Unassigned")
        .exists()
    )


def _student_queryset():
    User = get_user_model()
    return (
        User.objects.filter(Q(groups__name__startswith="Class: ") | Q(groups__name="Student"))
        .exclude(groups__name="Professor")
        .distinct()
    )


@login_required
def home(request):
    if _is_professor(request.user):
        return redirect("vip:professor_dashboard")
    elif _is_student(request.user):
        return redirect("vip:student_dashboard")
    else:
        return redirect("vip:dashboard")
    
@login_required
def dashboard(request):
    return render(request, "vip/dashboard.html")


@login_required
def account_settings(request):
    is_professor = _is_professor(request.user)
    status_message = ""
    error_message = ""
    password_form = PasswordChangeForm(request.user)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "change_password":
            password_form = PasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                status_message = "Password updated successfully."
            else:
                error_message = "Please fix password form errors."

        if action == "update_name" and is_professor:
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            request.user.first_name = first_name
            request.user.last_name = last_name
            request.user.save(update_fields=["first_name", "last_name"])
            status_message = "Profile name updated."

    return render(
        request,
        "vip/account_settings.html",
        {
            "is_professor": is_professor,
            "status_message": status_message,
            "error_message": error_message,
            "password_form": password_form,
        },
    )

# Professor views
@login_required
def professor_dashboard(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    current_tab = request.GET.get("tab", "prompts")
    if current_tab not in {"prompts", "logs", "test_chat"}:
        current_tab = "prompts"

    prompts = RolePrompt.objects.order_by("-updated_at")
    prompt_upload_form = PromptTextUploadForm()
    students = (
        _student_queryset().prefetch_related("groups")
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
            if group.name.startswith("Class: ") and group.name.lower() != "class: unassigned":
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
            "prompt_upload_form": prompt_upload_form,
            "students_by_class": students_by_class,
            "current_tab": current_tab,
        },
    )


@login_required
def professor_download_prompt_txt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    filename_base = slugify(prompt.title) or f"prompt-{prompt.id}"
    response = HttpResponse(prompt.content, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename_base}.txt"'
    return response


@login_required
def professor_download_prompt_template(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    template_path = _prompt_file_path("role_prompt_fillable.md")
    if not template_path:
        return HttpResponseBadRequest("Template file not found.")

    content = template_path.read_text(encoding="utf-8")
    response = HttpResponse(content, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="role_prompt_template.md"'
    return response


@login_required
@require_POST
def professor_upload_prompt_file(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    form = PromptTextUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")

    uploaded = form.cleaned_data["file"]
    try:
        content = uploaded.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return HttpResponseBadRequest("File must be UTF-8 text.")

    title = form.cleaned_data["title"].strip() if form.cleaned_data["title"] else Path(uploaded.name).stem
    parsed_initial = RolePromptForm.initial_from_content(
        content,
        title=title or "Uploaded Prompt",
        is_active=form.cleaned_data["is_active"],
    )
    structured_form = RolePromptForm(parsed_initial)
    if structured_form.is_valid():
        structured_content = structured_form.render_markdown_content()
    else:
        structured_content = content

    RolePrompt.objects.create(
        title=parsed_initial["title"] or "Uploaded Prompt",
        content=structured_content,
        created_by=request.user,
        is_active=form.cleaned_data["is_active"],
    )
    return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")


@login_required
def professor_manage_accounts(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    legacy_unassigned_group = Group.objects.filter(name__iexact="Class: Unassigned").first()
    if legacy_unassigned_group:
        legacy_unassigned_group.user_set.clear()
        legacy_unassigned_group.delete()

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
                selected_class = _normalize_class_name(single_form.cleaned_data["class_name"])
                class_group = None
                if selected_class:
                    class_group = Group.objects.filter(name=f"Class: {selected_class}").first()
                if selected_class and not class_group:
                    skipped_rows.append(
                        {"row": "Single form", "reason": "Please select an existing class group from the dropdown."}
                    )
                else:
                    account, error = _create_student_account(
                        first_name=single_form.cleaned_data["first_name"],
                        last_name=single_form.cleaned_data["last_name"],
                        net_id=single_form.cleaned_data["net_id"],
                        student_id=single_form.cleaned_data["student_id"],
                        class_name=selected_class,
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
                    "net_id": ["netid", "net_id", "username", "login"],
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
                    net_id = _extract_row_value(row, aliases["net_id"])
                    student_id = _extract_row_value(row, aliases["student_id"])
                    class_name = _extract_row_value(row, aliases["class_name"])
                    if re.match(r"^\d+\.0$", student_id):
                        student_id = student_id[:-2]

                    if not first_name or not last_name or not net_id or not student_id:
                        skipped_rows.append(
                            {
                                "row": idx,
                                "reason": "Missing required fields (first_name, last_name, net_id, student_id).",
                            }
                        )
                        continue
                    if not student_id.isdigit():
                        skipped_rows.append({"row": idx, "reason": "student_id must be numeric."})
                        continue

                    account, error = _create_student_account(
                        first_name=first_name,
                        last_name=last_name,
                        net_id=net_id,
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
                if not normalized_class:
                    skipped_rows.append({"row": "Create class", "reason": "Class name cannot be blank."})
                else:
                    Group.objects.get_or_create(name=f"Class: {normalized_class}")
                    info_message = f'Class group "{normalized_class}" is ready.'

        if action == "move_student":
            student_id = request.POST.get("student_id", "").strip()
            selected_class_group_id = request.POST.get("class_group_id", "").strip()
            student = get_object_or_404(_student_queryset(), pk=student_id)
            existing_class_groups = student.groups.filter(name__startswith="Class: ")
            student.groups.remove(*existing_class_groups)

            if selected_class_group_id == "__UNASSIGNED__":
                info_message = f"Moved {student.username} to Unassigned."
            elif selected_class_group_id:
                class_group = Group.objects.filter(pk=selected_class_group_id, name__startswith="Class: ").first()
                if class_group:
                    student.groups.add(class_group)
                    class_name = class_group.name.replace("Class: ", "", 1)
                    info_message = f"Moved {student.username} to class {class_name}."
                else:
                    skipped_rows.append({"row": "Move student", "reason": "Selected class group does not exist."})
            else:
                skipped_rows.append({"row": "Move student", "reason": "Please choose a class from the dropdown."})

        if action == "delete_student":
            student_id = request.POST.get("student_id", "").strip()
            student = get_object_or_404(_student_queryset(), pk=student_id)
            username = student.username
            student.delete()
            info_message = f"Deleted student account {username}."

        if action == "delete_class":
            class_group_id = request.POST.get("class_group_id", "").strip()
            class_group = Group.objects.filter(pk=class_group_id, name__startswith="Class: ").first()
            if class_group:
                class_name = class_group.name.replace("Class: ", "", 1)
                class_group.delete()
                info_message = f'Deleted class group "{class_name}".'
            else:
                skipped_rows.append({"row": "Delete class", "reason": "Class group not found."})

    class_groups = (
        Group.objects.filter(name__startswith="Class: ")
        .exclude(name__iexact="Class: Unassigned")
        .order_by("name")
    )
    roster_students = _student_queryset().prefetch_related("groups").order_by("username")
    student_rows = []
    for student in roster_students:
        class_names = [
            group.name.replace("Class: ", "", 1)
            for group in student.groups.all()
            if group.name.startswith("Class: ") and group.name.lower() != "class: unassigned"
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
    if not _is_professor(request.user):
        return redirect("vip:home")

    if request.method == "POST":
        form = RolePromptForm(request.POST)
        if form.is_valid():
            RolePrompt.objects.create(
                title=form.cleaned_data["title"],
                content=form.render_markdown_content(),
                created_by=request.user,
                is_active=form.cleaned_data["is_active"],
            )
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
    if not _is_professor(request.user):
        return redirect("vip:home")

    prompt = get_object_or_404(RolePrompt, pk=prompt_id)

    if request.method == "POST":
        form = RolePromptForm(request.POST)
        if form.is_valid():
            prompt.title = form.cleaned_data["title"]
            prompt.is_active = form.cleaned_data["is_active"]
            prompt.content = form.render_markdown_content()
            prompt.save(update_fields=["title", "is_active", "content", "updated_at"])
            return redirect("vip:professor_dashboard")
    else:
        form = RolePromptForm(initial=RolePromptForm.initial_from_prompt(prompt))

    return render(
        request,
        "vip/prompt_form.html",
        {"form": form, "page_title": "Edit Prompt"},
    )


@login_required
@require_POST
def set_active_prompt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    selected_prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    selected_prompt.is_active = True
    selected_prompt.save(update_fields=["is_active", "updated_at"])
    return redirect("vip:professor_dashboard")


@login_required
@require_POST
def deactivate_prompt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    selected_prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    selected_prompt.is_active = False
    selected_prompt.save(update_fields=["is_active", "updated_at"])
    return redirect("vip:professor_dashboard")


@login_required
@require_POST
def delete_prompt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    # Close any active sessions using this prompt before deletion so students
    # don't continue a conversation with a removed prompt.
    ChatSession.objects.filter(role_prompt=prompt, ended_at__isnull=True).update(ended_at=timezone.now())
    prompt.delete()
    return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")


@login_required
def professor_session_detail(request, session_id):
    if not _is_professor(request.user):
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
            "rendered_messages": _rendered_chat_messages(session, "Student"),
        },
    )


@login_required
def professor_student_logs(request, student_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    student = get_object_or_404(_student_queryset(), pk=student_id)
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
    if not _is_professor(request.user):
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
    if not _is_professor(request.user):
        return redirect("vip:home")

    student = get_object_or_404(_student_queryset(), pk=student_id)
    ChatSession.objects.filter(student=student).delete()
    return redirect("vip:professor_student_logs", student_id=student_id)


@login_required
@require_POST
def professor_reset_all_student_logs(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    student_ids = _student_queryset().values_list("id", flat=True)
    ChatSession.objects.filter(student_id__in=student_ids).delete()
    return redirect(f"{reverse('vip:professor_dashboard')}?tab=logs")


def _chat_url(view_name, selected_prompt=None, **params):
    query = {}
    if selected_prompt:
        query["prompt"] = selected_prompt.id
    query.update({key: value for key, value in params.items() if value is not None})
    url = reverse(view_name)
    return f"{url}?{urlencode(query)}" if query else url


def _rendered_chat_messages(session, user_label):
    if not session:
        return []

    rendered = []
    for message in session.messages.order_by("created_at"):
        display_content = message.content
        if message.sender == ChatMessage.Sender.ASSISTANT:
            display_content, _ = _parse_dialogue_and_emotion(message.content)
        rendered.append(
            {
                "id": message.id,
                "sender": message.sender,
                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else user_label,
                "created_at": message.created_at,
                "display_content": display_content,
            }
        )
    return rendered


def _chat_dashboard_context(
    active_prompts,
    selected_prompt,
    sessions,
    current_session,
    error_message,
    user_label,
    force_new,
):
    return {
        "active_prompts": active_prompts,
        "selected_prompt": selected_prompt,
        "sessions": sessions,
        "current_session": current_session,
        "rendered_messages": _rendered_chat_messages(current_session, user_label),
        "error_message": error_message,
        "force_new": force_new,
    }


def _selected_chat_state(request):
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

    current_session = None
    session_id = request.POST.get("session_id") or request.GET.get("session")
    if session_id:
        current_session = get_object_or_404(
            ChatSession.objects.filter(student=request.user).select_related("role_prompt"),
            pk=session_id,
        )
    elif request.GET.get("new") != "1" and selected_prompt:
        current_session = sessions.filter(role_prompt=selected_prompt).first()

    force_new = request.POST.get("force_new") == "1" or request.GET.get("force_new") == "1"
    return active_prompts, sessions, selected_prompt, current_session, force_new


def _usable_chat_session(user, selected_prompt, current_session, force_new):
    needs_session = (
        not current_session
        or current_session.role_prompt_id != selected_prompt.id
        or current_session.ended_at is not None
        or force_new
    )
    if needs_session:
        current_session = None
        if not force_new:
            current_session = (
                ChatSession.objects.filter(
                    student=user,
                    role_prompt=selected_prompt,
                    ended_at__isnull=True,
                )
                .order_by("-started_at")
                .first()
            )
        return current_session or ChatSession.objects.create(student=user, role_prompt=selected_prompt), None

    latest_message = current_session.messages.order_by("-created_at").first()
    if latest_message and latest_message.sender == ChatMessage.Sender.STUDENT:
        return current_session, "Please wait for the AI response before sending another message."
    return current_session, None


def _chat_dashboard(
    request,
    *,
    template_name,
    view_name,
    user_label,
    no_prompt_message,
    allow_delete_session=False,
):
    active_prompts, sessions, selected_prompt, current_session, force_new = _selected_chat_state(request)
    error_message = None

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "new_session":
            return redirect(_chat_url(view_name, selected_prompt, new=1, force_new=1))

        if allow_delete_session and action == "delete_session":
            target_session_id = request.POST.get("target_session_id") or request.POST.get("session_id")
            if target_session_id:
                get_object_or_404(ChatSession, pk=target_session_id, student=request.user).delete()
            return redirect(_chat_url(view_name, selected_prompt, new=1))

        if action == "close_conversation":
            if current_session and current_session.ended_at is None:
                current_session.ended_at = timezone.now()
                current_session.save(update_fields=["ended_at"])
            return redirect(_chat_url(view_name, selected_prompt, new=1))

        if action == "send_message":
            user_text = request.POST.get("message", "").strip()
            if not selected_prompt:
                error_message = no_prompt_message
            elif not user_text:
                error_message = "Please type a message before sending."
            else:
                current_session, error_message = _usable_chat_session(
                    request.user,
                    selected_prompt,
                    current_session,
                    force_new,
                )

                if not error_message:
                    completed_turns = current_session.messages.filter(
                        sender=ChatMessage.Sender.STUDENT
                    ).count()
                    if completed_turns >= MAX_TURNS:
                        error_message = "This conversation is already complete."

            if error_message:
                return render(
                    request,
                    template_name,
                    _chat_dashboard_context(
                        active_prompts,
                        selected_prompt,
                        sessions,
                        current_session,
                        error_message,
                        user_label,
                        force_new,
                    ),
                )

            existing_messages = current_session.messages.order_by("created_at")
            has_assistant_message = existing_messages.filter(sender=ChatMessage.Sender.ASSISTANT).exists()
            if not has_assistant_message:
                scenario = parse_scenario_prompt(selected_prompt.content)
                introduction = scenario.introduction
                if introduction:
                    ChatMessage.objects.create(
                        session=current_session,
                        sender=ChatMessage.Sender.ASSISTANT,
                        content=introduction,
                        voice_metadata=scenario.voice_metadata(introduction=True),
                    )

            ChatMessage.objects.create(
                session=current_session,
                sender=ChatMessage.Sender.STUDENT,
                content=user_text,
            )

            assistant_text, conversation_complete, debug_info = _generate_assistant_response(
                selected_prompt.content,
                current_session.messages.order_by("created_at"),
                current_session,
            )
            if assistant_text:
                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.ASSISTANT,
                    content=assistant_text,
                    voice_metadata=debug_info.get("voice_metadata", ""),
                )
            current_session.conversation_stage = debug_info.get(
                "current_stage", current_session.conversation_stage
            )
            current_session.conversation_phase = debug_info.get(
                "phase", current_session.conversation_phase
            )
            current_session.completion_status = conversation_complete
            current_session.save(
                update_fields=[
                    "conversation_stage",
                    "conversation_phase",
                    "completion_status",
                ]
            )
            logger.debug(
                "Chat response generated: user=%s session=%s phase=%s complete=%s reason=%s",
                request.user.username,
                current_session.id,
                debug_info.get("phase"),
                conversation_complete,
                debug_info.get("reason"),
            )
            if conversation_complete and current_session.ended_at is None:
                current_session.ended_at = timezone.now()
                current_session.save(update_fields=["ended_at"])
            return redirect(_chat_url(view_name, selected_prompt, session=current_session.id, autoplay=1))

    return render(
        request,
        template_name,
        _chat_dashboard_context(
            active_prompts,
            selected_prompt,
            sessions,
            current_session,
            error_message,
            user_label,
            force_new,
        ),
    )


@login_required
def professor_test_chat(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    return _chat_dashboard(
        request,
        template_name="vip/professor_test_chat.html",
        view_name="vip:professor_test_chat",
        user_label="Professor",
        no_prompt_message="No active prompt is available. Activate at least one prompt first.",
        allow_delete_session=True,
    )


# Student views

@login_required
def student_dashboard(request):
    if not _is_student(request.user):
        return redirect("vip:home")

    return _chat_dashboard(
        request,
        template_name="vip/student_dashboard.html",
        view_name="vip:student_dashboard",
        user_label="Student",
        no_prompt_message="No active prompt is available. Ask your professor to activate one.",
    )


@login_required
def student_download_session(request, session_id):
    if not (_is_student(request.user) or _is_professor(request.user)):
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
        content = message.content
        if message.sender == ChatMessage.Sender.ASSISTANT:
            content, _ = split_dialogue_and_voice(content)
        lines.append(f"[{message.created_at}] {speaker}: {content}")
        lines.append("")

    response = HttpResponse("\n".join(lines), content_type="text/plain")
    response["Content-Disposition"] = f'attachment; filename="chat_session_{session.id}.txt"'
    return response


@login_required
def student_message_tts(request, message_id):
    if not (_is_student(request.user) or _is_professor(request.user)):
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

    dialogue, embedded_emotion = split_dialogue_and_voice(message.content)
    emotion = getattr(message, "voice_metadata", "") or embedded_emotion
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
                input=dialogue or "No spoken dialogue.",
                instructions=emotion or "Speak naturally and clearly.",
            )
            return HttpResponse(response.read(), content_type="audio/mpeg")

        from gtts import gTTS

        clean_text = _clean_for_tts(dialogue)
        audio_buffer = io.BytesIO()
        tts = gTTS(text=clean_text or "No content", lang="en")
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return HttpResponse(audio_buffer.read(), content_type="audio/mpeg")
    except Exception as exc:
        return HttpResponseBadRequest(f"TTS error: {exc}")
