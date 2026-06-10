import os
import re
import io
import csv
import logging
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import Group
from django.contrib.auth import update_session_auth_hash
from django.db.models import Count, Max
from django.shortcuts import redirect
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
from .models import ChatMessage, ChatSession, ProfessorClass, ProfessorStudent, RolePrompt

"""
base
"""

logger = logging.getLogger(__name__)

DEFAULT_META_INSTRUCTIONS = (
    "Do not play both sides. Stay in character. "
    "Do not restart the introduction once conversation has begun."
)


def _split_markdown_sections(text):
    sections = {}
    current = ""
    for line in (text or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = _normalize_heading(match.group(1))
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _section_by_aliases(sections, aliases):
    normalized = {key: value for key, value in sections.items()}

    # Pass 1: exact match only.
    for alias in aliases:
        alias = _normalize_heading(alias)
        if alias in normalized:
            return normalized[alias]

    # Pass 2: ranked prefix match (prefer non-voice variants).
    best_value = ""
    best_score = None
    for alias in aliases:
        alias = _normalize_heading(alias)
        alias_tokens = alias.split()
        for key, value in normalized.items():
            key_tokens = key.split()
            if len(key_tokens) < len(alias_tokens):
                continue
            if key_tokens[: len(alias_tokens)] != alias_tokens:
                continue
            extra_tokens = key_tokens[len(alias_tokens) :]
            penalty = 5 if ("voice" in extra_tokens and "voice" not in alias_tokens) else 0
            score = len(extra_tokens) + penalty
            if best_score is None or score < best_score:
                best_score = score
                best_value = value
    return best_value


def _normalize_heading(text):
    value = re.sub(r"\(optional\)", "", (text or ""), flags=re.IGNORECASE)
    value = value.strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_template_prefix():
    prompt_template_path = Path(settings.BASE_DIR).parent / "prompts" / "prompt_template.md"
    if prompt_template_path.exists():
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
    else:
        prompt_template = ""
    if "{role}" in prompt_template:
        return prompt_template.split("{role}", 1)[0].strip()
    return prompt_template.strip()


def _parse_role_config(role_text):
    sections = _split_markdown_sections(role_text)
    role = _section_by_aliases(sections, ["role", "role summary", "character"])
    learner_role = _section_by_aliases(sections, ["learner role", "user role"])
    voice_gender = (_section_by_aliases(sections, ["voice gender", "voice"]) or "female").lower().strip()
    if voice_gender not in {"male", "female"}:
        merged = f"{role_text}\n{_section_by_aliases(sections, ['introduction'])}"
        voice_gender = "male" if "male voice" in merged.lower() else "female"
    voice_style = _section_by_aliases(sections, ["voice style", "voice instructions"]) or "speak naturally and clearly"
    intro_voice_gender = (
        _section_by_aliases(sections, ["introduction voice gender", "intro voice gender"]) or voice_gender
    ).lower().strip()
    if intro_voice_gender not in {"male", "female"}:
        intro_voice_gender = voice_gender
    intro_voice_style = (
        _section_by_aliases(sections, ["introduction voice style", "intro voice style"]) or voice_style
    )
    return {
        "role": role or role_text.strip(),
        "learner_role": learner_role or "nursing student",
        "voice_gender": voice_gender,
        "voice_style": voice_style.strip(),
        "intro_voice_gender": intro_voice_gender,
        "intro_voice_style": intro_voice_style.strip(),
        "introduction": _section_by_aliases(sections, ["introduction", "introduction: greeting"]),
        "opening_line": _section_by_aliases(sections, ["opening line"]),
        "beginning": _section_by_aliases(sections, ["beginning", "conversation progression: beginning"]),
        "middle": _section_by_aliases(sections, ["middle", "conversation progression: middle"]),
        "ending": _section_by_aliases(sections, ["ending", "end", "conversation progression: end"]),
        "closing": _section_by_aliases(sections, ["closing", "final response"]),
        "meta": _section_by_aliases(sections, ["meta instructions", "meta instruction", "meta-instructions", "notes"]),
        "begin_cues": _section_by_aliases(
            sections,
            ["beginning to middle cues", "begin-to-middle cues", "middle trigger", "middle triggers", "trigger"],
        ),
        "end_cues": _section_by_aliases(
            sections,
            ["middle to ending cues", "middle-to-ending cues", "ending trigger", "ending triggers"],
        ),
    }


def _parse_cues(text):
    if not text:
        return []
    cues = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip().lower()
        if not line:
            continue
        if "," in line:
            cues.extend([part.strip() for part in line.split(",") if part.strip()])
        else:
            cues.append(line)
    return cues


def _enforce_voice_format(text, voice_gender, voice_style):
    text = (text or "").strip()
    if not text:
        return f"[{voice_gender} voice, {voice_style}]"
    if "[" not in text or "]" not in text:
        return f"{text}\n\n[{voice_gender} voice, {voice_style}]"
    found = re.findall(r"\[([^\]]+)\]", text, flags=re.DOTALL)
    if not found:
        return f"{text}\n\n[{voice_gender} voice, {voice_style}]"
    last = found[-1].lower()
    if "male voice" not in last and "female voice" not in last:
        return re.sub(
            r"\[([^\]]+)\]\s*$",
            f"[{voice_gender} voice, {found[-1].strip()}]",
            text,
            count=1,
            flags=re.DOTALL,
        )
    return text


def _conversation_history_text(session_messages):
    conversation_lines = []
    for message in session_messages:
        label = "Student" if message.sender == ChatMessage.Sender.STUDENT else "Assistant"
        conversation_lines.append(f"{label}: {message.content}")
    return "\n".join(conversation_lines).strip()


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
    match = re.match(r"^(.*?)(\[.*\])?$", text.strip(), re.DOTALL)
    if not match:
        return text, ""
    dialogue = match.group(1).strip()
    emotion = match.group(2).strip("[]") if match.group(2) else ""
    return dialogue, emotion


def _clean_for_tts(text):
    return re.sub(r"\([^)]*\)", "", text).strip()


def _normalize_for_close_match(text):
    value = (text or "").lower()
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _generate_assistant_response(role_text, session_messages):
    api_key = _load_api_key_from_txt()
    if not api_key:
        return "OpenAI API key is not configured on the server yet.", False, {"stage": "error", "reason": "missing_api_key"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        config = _parse_role_config(role_text)
        meta_instructions = (config.get("meta") or "").strip() or DEFAULT_META_INSTRUCTIONS
        template_prefix = _extract_template_prefix()
        history = _conversation_history_text(session_messages)
        assistant_count = sum(1 for m in session_messages if m.sender == ChatMessage.Sender.ASSISTANT)
        last_user = ""
        for message in reversed(session_messages):
            if message.sender == ChatMessage.Sender.STUDENT:
                last_user = message.content.strip().lower()
                break

        close_phrases = [
            "thank you",
            "thanks",
            "goodbye",
            "bye",
            "that is all",
            "no that's all",
            "no, that's all",
            "nothing else",
            "we're done",
            "that covers everything",
        ]
        should_close = any(p in last_user for p in close_phrases)
        begin_to_middle_cues = _parse_cues(config["begin_cues"]) or [
            "home",
            "medication",
            "medicine",
            "daily",
            "routine",
            "help",
            "support",
            "safety",
            "family",
            "caregiver",
            "hospice",
            "feeding",
            "pain",
        ]
        middle_to_ending_cues = _parse_cues(config["end_cues"]) or [
            "anything else",
            "before we finish",
            "summarize",
            "wrap up",
            "closing",
            "goodbye",
            "thanks",
            "stop conversation",
            "end conversation",
            "finish this conversation",
        ]
        ending_intent_phrases = [
            "i want to stop",
            "can we stop",
            "let's stop",
            "i want to end this",
            "end this conversation",
            "stop this conversation",
            "we can end here",
            "i think we're done for now",
        ]

        user_history = " ".join(
            m.content.strip().lower() for m in session_messages if m.sender == ChatMessage.Sender.STUDENT
        )
        has_middle_signal = any(c in user_history for c in begin_to_middle_cues)
        has_ending_signal = any(c in user_history for c in middle_to_ending_cues)
        has_ending_intent = any(p in last_user for p in ending_intent_phrases)

        stage = "beginning"
        if assistant_count == 0:
            stage = "intro"
        elif has_ending_signal or has_ending_intent:
            stage = "ending"
        elif has_middle_signal:
            stage = "middle"

        if should_close:
            stage = "closing"

        if stage == "intro":
            intro = config["introduction"] or "Hello, I am ready to begin this roleplay."
            return _enforce_voice_format(
                intro,
                config["intro_voice_gender"],
                config["intro_voice_style"],
            ), False, {"stage": stage, "reason": "first_assistant_turn"}

        if stage == "beginning" and assistant_count == 1 and config["opening_line"]:
            return _enforce_voice_format(
                config["opening_line"],
                config["voice_gender"],
                config["voice_style"],
            ), False, {"stage": stage, "reason": "hardcoded_opening_line"}

        if stage == "closing":
            closing = config["closing"] or (
                "Thank you for this conversation. I appreciate your help today."
            )
            return _enforce_voice_format(closing, config["voice_gender"], config["voice_style"]), True, {
                "stage": stage,
                "reason": "closing_stage_selected",
            }

        stage_instructions = config.get(stage, "") or config["middle"] or config["role"]
        system_prompt = (
            f"{template_prefix}\n\n"
            "You are a standardized roleplay participant in a structured state-machine conversation.\n"
            "Do not play both sides.\n"
            "Do not restart introduction.\n"
            f"Learner role: {config['learner_role']}\n"
            f"Voice must be bracketed with '{config['voice_gender']} voice'.\n"
            f"Meta instructions:\n{meta_instructions}\n"
            "Output format:\n"
            "Dialogue\n\n"
            f"[{config['voice_gender']} voice, style instructions]"
        )
        user_prompt = (
            f"Character profile:\n{config['role']}\n\n"
            f"Current stage: {stage}\n"
            f"Stage instructions:\n{stage_instructions}\n\n"
            f"Conversation so far:\n{history}\n\n"
            "Respond in character."
        )
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.output_text or "The assistant returned an empty response."
        assistant_text = _enforce_voice_format(text, config["voice_gender"], config["voice_style"])

        # Auto-close once the assistant has delivered the concluding line/content.
        assistant_dialogue, _ = _parse_dialogue_and_emotion(assistant_text)
        assistant_norm = _normalize_for_close_match(assistant_dialogue)
        configured_closing_norm = _normalize_for_close_match(config.get("closing", ""))
        close_markers = [
            "thank you for this conversation",
            "thank you for engaging with virtual conversation simulation",
            "please remember to download your conversation record",
            "goodbye",
            "take care",
        ]
        content_signals_close = False
        if configured_closing_norm and (
            configured_closing_norm in assistant_norm or assistant_norm in configured_closing_norm
        ):
            content_signals_close = True
        elif any(marker in assistant_norm for marker in close_markers):
            content_signals_close = True

        should_auto_close = should_close or (stage in {"ending", "closing"} and content_signals_close)
        return assistant_text, should_auto_close, {
            "stage": stage,
            "assistant_count": assistant_count,
            "has_middle_signal": has_middle_signal,
            "has_ending_signal": has_ending_signal,
            "has_ending_intent": has_ending_intent,
            "user_requested_close": should_close,
            "content_signals_close": content_signals_close,
            "reason": "late_stage_close_signal" if should_auto_close else "continue",
        }
    except Exception as exc:
        return f"Assistant error: {exc}", False, {"stage": "error", "reason": str(exc)}


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


def _get_owned_class(professor, class_name):
    normalized_class = _normalize_class_name(class_name)
    if not normalized_class:
        return None
    return ProfessorClass.objects.filter(professor=professor, name=normalized_class).first()


def _create_student_account(first_name, last_name, net_id, student_id, class_name, professor, create_class=False):
    User = get_user_model()
    username = _normalize_net_id(net_id)
    password = _build_student_password(first_name, last_name, student_id)
    normalized_class = _normalize_class_name(class_name)
    class_group = None

    if not username:
        return None, "NetID is required."
    if User.objects.filter(username=username).exists():
        return None, f'NetID "{username}" already exists.'
    if normalized_class:
        if create_class:
            class_group, _ = ProfessorClass.objects.get_or_create(
                professor=professor,
                name=normalized_class,
            )
        else:
            class_group = _get_owned_class(professor, normalized_class)
        if not class_group:
            return None, "Please select an existing class created under your account."

    user = User.objects.create_user(
        username=username,
        password=password,
        first_name=first_name.strip(),
        last_name=last_name.strip(),
    )

    student_group, _ = Group.objects.get_or_create(name="Student")
    user.groups.add(student_group)
    ProfessorStudent.objects.create(
        professor=professor,
        student=user,
        class_group=class_group,
        student_number=student_id.strip(),
    )
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
    return user.groups.filter(name__iexact="Student").exists() or hasattr(user, "professor_assignment")


def _student_queryset(professor):
    User = get_user_model()
    return User.objects.filter(professor_assignment__professor=professor).distinct()


def _student_assignment_queryset(professor):
    return ProfessorStudent.objects.filter(professor=professor).select_related("student", "class_group")


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

"""
PROFESSOR
"""
@login_required
def professor_dashboard(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    current_tab = request.GET.get("tab", "prompts")
    if current_tab not in {"prompts", "logs", "test_chat"}:
        current_tab = "prompts"

    prompts = RolePrompt.objects.order_by("-updated_at")
    prompt_upload_form = PromptTextUploadForm()
    student_assignments = (
        _student_assignment_queryset(request.user)
        .annotate(
            session_count=Count("student__chat_sessions", distinct=True),
            last_session_at=Max("student__chat_sessions__started_at"),
        )
        .order_by("class_group__name", "student__username")
    )
    students_by_class = {}
    for assignment in student_assignments:
        student = assignment.student
        student.session_count = assignment.session_count
        student.last_session_at = assignment.last_session_at
        class_name = assignment.class_group.name if assignment.class_group else "Unassigned"
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

    template_path = Path(settings.BASE_DIR).parent / "prompts" / "role_prompt_fillable.md"
    if not template_path.exists():
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
                account, error = _create_student_account(
                    first_name=single_form.cleaned_data["first_name"],
                    last_name=single_form.cleaned_data["last_name"],
                    net_id=single_form.cleaned_data["net_id"],
                    student_id=single_form.cleaned_data["student_id"],
                    class_name=selected_class,
                    professor=request.user,
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
                        professor=request.user,
                        create_class=True,
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
                    ProfessorClass.objects.get_or_create(professor=request.user, name=normalized_class)
                    info_message = f'Class "{normalized_class}" is ready.'

        if action == "move_student":
            student_id = request.POST.get("student_id", "").strip()
            selected_class_group_id = request.POST.get("class_group_id", "").strip()
            assignment = get_object_or_404(_student_assignment_queryset(request.user), student_id=student_id)
            student = assignment.student

            if selected_class_group_id == "__UNASSIGNED__":
                assignment.class_group = None
                assignment.save(update_fields=["class_group"])
                info_message = f"Moved {student.username} to Unassigned."
            elif selected_class_group_id:
                class_group = ProfessorClass.objects.filter(pk=selected_class_group_id, professor=request.user).first()
                if class_group:
                    assignment.class_group = class_group
                    assignment.save(update_fields=["class_group"])
                    info_message = f"Moved {student.username} to class {class_group.name}."
                else:
                    skipped_rows.append({"row": "Move student", "reason": "Selected class does not exist."})
            else:
                skipped_rows.append({"row": "Move student", "reason": "Please choose a class from the dropdown."})

        if action == "delete_student":
            student_id = request.POST.get("student_id", "").strip()
            assignment = get_object_or_404(_student_assignment_queryset(request.user), student_id=student_id)
            student = assignment.student
            username = student.username
            student.delete()
            info_message = f"Deleted student account {username}."

        if action == "delete_class":
            class_group_id = request.POST.get("class_group_id", "").strip()
            class_group = ProfessorClass.objects.filter(pk=class_group_id, professor=request.user).first()
            if class_group:
                class_name = class_group.name
                class_group.delete()
                info_message = f'Deleted class "{class_name}".'
            else:
                skipped_rows.append({"row": "Delete class", "reason": "Class not found."})

    class_groups = ProfessorClass.objects.filter(professor=request.user).order_by("name")
    roster_students = _student_assignment_queryset(request.user).order_by("student__username")
    student_rows = []
    for assignment in roster_students:
        student = assignment.student
        student_rows.append(
            {
                "id": student.id,
                "username": student.username,
                "first_name": student.first_name,
                "last_name": student.last_name,
                "classes_display": assignment.class_group.name if assignment.class_group else "Unassigned",
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
        ChatSession.objects.filter(student__professor_assignment__professor=request.user).select_related(
            "student",
            "role_prompt",
        ),
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
    if not _is_professor(request.user):
        return redirect("vip:home")

    student = get_object_or_404(_student_queryset(request.user), pk=student_id)
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

    session = get_object_or_404(
        ChatSession.objects.filter(student__professor_assignment__professor=request.user),
        pk=session_id,
    )
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

    student = get_object_or_404(_student_queryset(request.user), pk=student_id)
    ChatSession.objects.filter(student=student).delete()
    return redirect("vip:professor_student_logs", student_id=student_id)


@login_required
@require_POST
def professor_reset_all_student_logs(request):
    if not _is_professor(request.user):
        return redirect("vip:home")

    student_ids = _student_queryset(request.user).values_list("id", flat=True)
    ChatSession.objects.filter(student_id__in=student_ids).delete()
    return redirect(f"{reverse('vip:professor_dashboard')}?tab=logs")


@login_required
def professor_test_chat(request):
    if not _is_professor(request.user):
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
            target = reverse("vip:professor_test_chat")
            if selected_prompt:
                target += f"?prompt={selected_prompt.id}&new=1&force_new=1"
            return redirect(target)

        if action == "delete_session":
            target_session_id = request.POST.get("target_session_id") or request.POST.get("session_id")
            if target_session_id:
                session_to_delete = get_object_or_404(ChatSession, pk=target_session_id, student=request.user)
                session_to_delete.delete()
            target = reverse("vip:professor_test_chat")
            if selected_prompt:
                target += f"?prompt={selected_prompt.id}&new=1"
            return redirect(target)

        if action == "close_conversation":
            if current_session and current_session.ended_at is None:
                current_session.ended_at = timezone.now()
                current_session.save(update_fields=["ended_at"])
            target = reverse("vip:professor_test_chat")
            if selected_prompt:
                target += f"?prompt={selected_prompt.id}&new=1"
            return redirect(target)

        if action == "send_message":
            user_text = request.POST.get("message", "").strip()
            if not selected_prompt:
                error_message = "No active prompt is available. Activate at least one prompt first."
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
                                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else "Professor",
                                "created_at": message.created_at,
                                "display_content": display_content,
                            }
                        )
                    return render(
                        request,
                        "vip/professor_test_chat.html",
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

                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.STUDENT,
                    content=user_text,
                )

                session_messages = current_session.messages.order_by("created_at")
                assistant_text, should_auto_close, debug_info = _generate_assistant_response(
                    selected_prompt.content,
                    session_messages,
                )
                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.ASSISTANT,
                    content=assistant_text,
                )
                logger.debug(
                    "Chat response generated: user=%s session=%s stage=%s auto_close=%s reason=%s",
                    request.user.username,
                    current_session.id,
                    debug_info.get("stage"),
                    should_auto_close,
                    debug_info.get("reason"),
                )
                if should_auto_close and current_session.ended_at is None:
                    current_session.ended_at = timezone.now()
                    current_session.save(update_fields=["ended_at"])
                return redirect(
                    f"{reverse('vip:professor_test_chat')}?prompt={selected_prompt.id}&session={current_session.id}&autoplay=1"
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
                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else "Professor",
                "created_at": message.created_at,
                "display_content": display_content,
            }
        )

    return render(
        request,
        "vip/professor_test_chat.html",
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


"""
STUDENT
"""

@login_required
def student_dashboard(request):
    if not _is_student(request.user):
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
                assistant_text, should_auto_close, debug_info = _generate_assistant_response(
                    selected_prompt.content,
                    session_messages,
                )
                ChatMessage.objects.create(
                    session=current_session,
                    sender=ChatMessage.Sender.ASSISTANT,
                    content=assistant_text,
                )
                logger.debug(
                    "Chat response generated: user=%s session=%s stage=%s auto_close=%s reason=%s",
                    request.user.username,
                    current_session.id,
                    debug_info.get("stage"),
                    should_auto_close,
                    debug_info.get("reason"),
                )
                if should_auto_close and current_session.ended_at is None:
                    current_session.ended_at = timezone.now()
                    current_session.save(update_fields=["ended_at"])
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
        lines.append(f"[{message.created_at}] {speaker}: {message.content}")
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
