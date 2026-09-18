import os
import re
import io
import csv
import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.core.exceptions import ValidationError
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import Group
from django.contrib.auth import update_session_auth_hash
from django.db import transaction
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
from .models import ChatMessage, ChatSession, RolePrompt, VoiceConversation
from .conversation_graph import MAX_TURNS
from .conversation_engine import (
    ConversationEngine,
    clean_dialogue_for_display,
    split_dialogue_and_voice,
)
from .conversation_scenario import parse_scenario_prompt
from .speech_engine.service import (
    ElevenLabsSpeechStream,
    SpeechEngineConfigurationError,
    api_key as elevenlabs_api_key,
    is_speech_engine_configured,
    issue_webrtc_token,
    list_elevenlabs_voice_options,
    speech_engine_max_duration_seconds,
)

logger = logging.getLogger(__name__)


def _prompt_voice_options():
    """Return form choices and browser-safe provider preview URLs together."""
    options = list_elevenlabs_voice_options()
    return (
        [(option.voice_id, option.name) for option in options],
        {
            option.voice_id: option.preview_url
            for option in options
            if option.preview_url
        },
    )

def _prompt_file_path(filename):
    candidates = [
        Path(settings.BASE_DIR).parent / "prompts" / filename,
        Path(settings.BASE_DIR) / "prompts" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def _extract_template_prefix():
    global_prompt_path = _prompt_file_path("global_prompt.md")
    if global_prompt_path:
        return global_prompt_path.read_text(encoding="utf-8").strip()

    # Keep compatibility with legacy deployments that still provide the old
    # template file, but never prefer it over the global prompt.
    prompt_template_path = _prompt_file_path("prompt_template.md")
    if not prompt_template_path:
        return ""
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    if "{role}" in prompt_template:
        return prompt_template.split("{role}", 1)[0].strip()
    return prompt_template.strip()


def _load_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return api_key


def _clean_for_tts(text):
    return re.sub(r"\([^)]*\)", "", text).strip()


def _generate_assistant_response(
    role_text,
    session_messages,
    session=None,
):
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    engine = ConversationEngine(_extract_template_prefix(), api_key)
    conversation_state = None
    if session is not None:
        conversation_state = {
            "core_question_state": session.core_question_state,
            "completion_status": session.completion_status,
            "current_stage": session.conversation_stage,
            "active_objective": session.active_objective,
            "covered_objectives": list(session.covered_objectives or []),
            "unresolved_objectives": list(session.unresolved_objectives or []),
            "recent_topics": list(session.recent_topics or []),
            "active_topic": session.active_topic,
            "covered_topics": list(session.covered_topics or []),
            "unresolved_topics": list(session.unresolved_topics or []),
            "topic_turn_counts": dict(session.topic_turn_counts or {}),
            "ending_ready": session.ending_ready,
        }
    return engine.respond(
        role_text,
        list(session_messages),
        conversation_state=conversation_state,
    )


def _request_turn_id(value):
    """Accept a safe client retry ID or create one for a non-voice submit."""
    candidate = (value or "").strip()
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate):
        return candidate
    return uuid.uuid4().hex


def _accept_learner_turn(session, user_text, turn_id, *, allow_voice_supersede=False):
    """Atomically claim one learner turn for a session.

    The session row is the lock. A retry with the same ID is idempotent; a
    different request cannot add a second pending learner message while the
    first response is being generated.
    """
    with transaction.atomic():
        locked_session = ChatSession.objects.select_for_update().get(pk=session.pk)
        existing_student = locked_session.messages.filter(
            sender=ChatMessage.Sender.STUDENT,
            turn_id=turn_id,
        ).first()
        if locked_session.ended_at is not None:
            return locked_session, "closed", None
        if locked_session.active_turn_id and (
            locked_session.active_claimed_at is None
            or locked_session.active_claimed_at < timezone.now() - timedelta(seconds=90)
        ):
            locked_session.active_turn_id = ""
            locked_session.active_claim_id = ""
            locked_session.active_claimed_at = None
            locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
        if existing_student:
            if existing_student.content.strip() != user_text.strip():
                logger.warning(
                    "Rejecting reused turn ID with different content session=%s turn=%s",
                    session.pk,
                    turn_id,
                )
                return locked_session, "conflict", None
            existing_assistant = locked_session.messages.filter(
                sender=ChatMessage.Sender.ASSISTANT,
                turn_id=turn_id,
            ).first()
            if existing_assistant or locked_session.last_completed_turn_id == turn_id:
                return locked_session, "duplicate", existing_assistant
            if locked_session.active_turn_id:
                return locked_session, "pending", None

            # A failed/cancelled request leaves its learner message in the
            # transcript so a retry preserves history. Reclaim that same
            # message instead of inserting a second learner turn.
            locked_session.active_turn_id = turn_id
            locked_session.active_claim_id = uuid.uuid4().hex
            locked_session.active_claimed_at = timezone.now()
            locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
            return locked_session, "accepted", None

        latest = locked_session.messages.order_by("-id").first()
        if locked_session.active_turn_id:
            if (
                allow_voice_supersede
                and latest
                and latest.sender == ChatMessage.Sender.STUDENT
                and latest.turn_id != turn_id
            ):
                # Speech Engine cancels an in-flight callback when it delivers
                # a replacement finalized event.  That cancellation and its
                # database release run on different async/sync boundaries, so
                # the replacement can legitimately arrive first.  Give the
                # newer provider event the claim now.  A late worker for the
                # old claim cannot persist because its claim ID no longer
                # matches.
                logger.info(
                    "Superseding interrupted voice turn session=%s old_turn=%s new_turn=%s",
                    locked_session.pk,
                    locked_session.active_turn_id,
                    turn_id,
                )
                latest.content = user_text
                latest.turn_id = turn_id
                latest.save(update_fields=["content", "turn_id"])
                locked_session.active_turn_id = turn_id
                locked_session.active_claim_id = uuid.uuid4().hex
                locked_session.active_claimed_at = timezone.now()
                locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
                return locked_session, "accepted", None
            return locked_session, "pending", None

        if latest and latest.sender == ChatMessage.Sender.STUDENT:
            if allow_voice_supersede and latest.turn_id != turn_id:
                # ElevenLabs can finalize a replacement transcript while the
                # previous callback is being interrupted. The old learner row
                # has no assistant pair, so replace it rather than leaving
                # the new provider event pending forever or creating two
                # learner turns for one utterance.
                latest.content = user_text
                latest.turn_id = turn_id
                latest.save(update_fields=["content", "turn_id"])
                locked_session.active_turn_id = turn_id
                locked_session.active_claim_id = uuid.uuid4().hex
                locked_session.active_claimed_at = timezone.now()
                locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
                return locked_session, "accepted", None
            return locked_session, "pending", None

        ChatMessage.objects.create(
            session=locked_session,
            sender=ChatMessage.Sender.STUDENT,
            content=user_text,
            turn_id=turn_id,
        )
        locked_session.active_turn_id = turn_id
        locked_session.active_claim_id = uuid.uuid4().hex
        locked_session.active_claimed_at = timezone.now()
        locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
        return locked_session, "accepted", None


def _release_learner_turn(session, turn_id, claim_id=""):
    """Release a failed/cancelled claim without deleting the learner row."""
    if not turn_id:
        return False
    with transaction.atomic():
        locked_session = ChatSession.objects.select_for_update().get(pk=session.pk)
        if locked_session.active_turn_id != turn_id:
            return False
        if claim_id and locked_session.active_claim_id != claim_id:
            return False
        locked_session.active_turn_id = ""
        locked_session.active_claim_id = ""
        locked_session.active_claimed_at = None
        locked_session.save(update_fields=["active_turn_id", "active_claim_id", "active_claimed_at"])
    return True


def _persist_assistant_response(
    session,
    assistant_text,
    conversation_complete,
    debug_info,
    turn_id="",
    cancellation_callback=None,
    claim_id="",
):
    """Commit at most one response, and never commit for a stale turn."""
    with transaction.atomic():
        locked_session = ChatSession.objects.select_for_update().get(pk=session.pk)
        if turn_id:
            if locked_session.ended_at is not None:
                return False
            if cancellation_callback and cancellation_callback():
                logger.info("Ignoring cancelled assistant response session=%s turn=%s", session.pk, turn_id)
                return False
            existing = locked_session.messages.filter(
                sender=ChatMessage.Sender.ASSISTANT,
                turn_id=turn_id,
            ).first()
            if existing:
                return False
            if locked_session.active_turn_id != turn_id:
                logger.warning("Ignoring unclaimed assistant response session=%s turn=%s", session.pk, turn_id)
                return False
            if claim_id and locked_session.active_claim_id != claim_id:
                logger.warning("Ignoring stale assistant claim session=%s turn=%s", session.pk, turn_id)
                return False
            latest = locked_session.messages.order_by("-id").first()
            if not latest or latest.sender != ChatMessage.Sender.STUDENT or latest.turn_id != turn_id:
                logger.warning("Ignoring stale assistant response session=%s turn=%s", session.pk, turn_id)
                return False

        if assistant_text:
            ChatMessage.objects.create(
                session=locked_session,
                sender=ChatMessage.Sender.ASSISTANT,
                content=assistant_text,
                voice_metadata=debug_info.get("voice_metadata", ""),
                turn_id=turn_id,
            )
        locked_session.conversation_stage = debug_info.get("current_stage", locked_session.conversation_stage)
        locked_session.conversation_phase = debug_info.get("phase", locked_session.conversation_phase)
        locked_session.completion_status = conversation_complete
        locked_session.active_objective = debug_info.get("active_objective", locked_session.active_objective)
        covered_objectives = debug_info.get("covered_objectives")
        if isinstance(covered_objectives, list):
            locked_session.covered_objectives = covered_objectives
        unresolved_objectives = debug_info.get("unresolved_objectives")
        if isinstance(unresolved_objectives, list):
            locked_session.unresolved_objectives = unresolved_objectives
        recent_topics = debug_info.get("recent_topics")
        if isinstance(recent_topics, list):
            locked_session.recent_topics = recent_topics
        locked_session.active_topic = debug_info.get("active_topic", locked_session.active_topic)
        covered_topics = debug_info.get("covered_topics")
        if isinstance(covered_topics, list):
            locked_session.covered_topics = covered_topics
        unresolved_topics = debug_info.get("unresolved_topics")
        if isinstance(unresolved_topics, list):
            locked_session.unresolved_topics = unresolved_topics
        topic_turn_counts = debug_info.get("topic_turn_counts")
        if isinstance(topic_turn_counts, dict):
            locked_session.topic_turn_counts = topic_turn_counts
        if "ending_ready" in debug_info:
            locked_session.ending_ready = bool(debug_info["ending_ready"])
        update_fields = [
            "conversation_stage",
            "conversation_phase",
            "completion_status",
            "active_objective",
            "covered_objectives",
            "unresolved_objectives",
            "recent_topics",
            "active_topic",
            "covered_topics",
            "unresolved_topics",
            "topic_turn_counts",
            "ending_ready",
        ]
        if turn_id:
            locked_session.active_turn_id = ""
            locked_session.active_claim_id = ""
            locked_session.active_claimed_at = None
            locked_session.last_completed_turn_id = turn_id
            update_fields.extend(["active_turn_id", "active_claim_id", "active_claimed_at", "last_completed_turn_id"])
        if "core_question_state" in debug_info:
            locked_session.core_question_state = debug_info["core_question_state"]
            update_fields.append("core_question_state")
        if conversation_complete and locked_session.ended_at is None:
            locked_session.ended_at = timezone.now()
            update_fields.append("ended_at")
        locked_session.save(update_fields=update_fields)
    return True


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


def _can_use_voice_practice(user):
    return _is_student(user) or _is_professor(user)


def _deactivate_incomplete_prompts(prompts):
    """Keep legacy/bulk-created incomplete prompts from remaining active."""
    incomplete_ids = []
    for prompt in prompts:
        if prompt.is_active and not prompt.is_complete:
            prompt.is_active = False
            incomplete_ids.append(prompt.pk)
    if incomplete_ids:
        RolePrompt.objects.filter(pk__in=incomplete_ids, is_active=True).update(is_active=False)
    return prompts


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

    prompts = _deactivate_incomplete_prompts(
        list(RolePrompt.objects.order_by("-updated_at"))
    )
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

    template_path = _prompt_file_path("role_prompt.md")
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
    voice_choices, _voice_preview_urls = _prompt_voice_options()
    structured_form = RolePromptForm(parsed_initial, voice_choices=voice_choices)
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
    messages.success(request, "Your prompt has been saved")
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

    voice_choices, voice_preview_urls = _prompt_voice_options()
    if request.method == "POST":
        form = RolePromptForm(request.POST, voice_choices=voice_choices)
        if form.is_valid():
            RolePrompt.objects.create(
                title=form.cleaned_data["title"],
                content=form.render_markdown_content(),
                created_by=request.user,
                is_active=form.cleaned_data["is_active"],
            )
            messages.success(request, "Your prompt has been saved")
            return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")
    else:
        form = RolePromptForm(voice_choices=voice_choices)

    return render(
        request,
        "vip/prompt_form.html",
        {
            "form": form,
            "page_title": "Add Prompt",
            "voice_preview_urls": voice_preview_urls,
        },
    )

@login_required
def edit_prompt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    if prompt.is_active and not prompt.is_complete:
        prompt.is_active = False
        prompt.save(update_fields=["is_active", "updated_at"])

    voice_choices, voice_preview_urls = _prompt_voice_options()
    if request.method == "POST":
        form = RolePromptForm(request.POST, voice_choices=voice_choices)
        if form.is_valid():
            prompt.title = form.cleaned_data["title"]
            prompt.is_active = form.cleaned_data["is_active"]
            prompt.content = form.render_markdown_content()
            prompt.save(update_fields=["title", "is_active", "content", "updated_at"])
            messages.success(request, "Your prompt has been saved")
            return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")
        if prompt.is_active:
            prompt.is_active = False
            prompt.save(update_fields=["is_active", "updated_at"])
    else:
        form = RolePromptForm(
            initial=RolePromptForm.initial_from_prompt(prompt),
            voice_choices=voice_choices,
        )

    return render(
        request,
        "vip/prompt_form.html",
        {
            "form": form,
            "page_title": "Edit Prompt",
            "voice_preview_urls": voice_preview_urls,
        },
    )


@login_required
@require_POST
def set_active_prompt(request, prompt_id):
    if not _is_professor(request.user):
        return redirect("vip:home")

    selected_prompt = get_object_or_404(RolePrompt, pk=prompt_id)
    if not selected_prompt.is_complete:
        selected_prompt.is_active = False
        selected_prompt.save(update_fields=["is_active", "updated_at"])
        messages.error(
            request,
            "Complete every required prompt field before activating this prompt.",
        )
        return redirect(f"{reverse('vip:professor_dashboard')}?tab=prompts")
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
            "simulation_introduction": _simulation_introduction(session=session),
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


def _scenario_text_for_session(session):
    if not session:
        return ""
    return session.scenario_content or getattr(session.role_prompt, "content", "")


def _simulation_introduction(session=None, prompt=None):
    role_text = _scenario_text_for_session(session) if session else getattr(prompt, "content", "")
    if not role_text:
        return "Welcome to the virtual patient simulation. When you're ready, introduce yourself and begin the conversation."
    scenario = parse_scenario_prompt(role_text)
    if scenario.introduction.strip():
        return scenario.introduction.strip()
    character = _role_summary(scenario.character, "the simulated character")
    learner = _role_summary(scenario.learner, "the learner in this simulation")
    return (
        f"Welcome to the virtual patient simulation. You are speaking with {character}. "
        f"Your role is {learner}. When you're ready, introduce yourself and begin the conversation."
    )


def _role_summary(value, fallback):
    value = re.sub(r"^\s*(?:you are|the human participant (?:is|plays))\s+", "", value or "", flags=re.I)
    first_sentence = re.split(r"(?<=[.!?])\s+|\n", value.strip(), maxsplit=1)[0].strip().rstrip(".!?")
    # Character prompts commonly put the name before a descriptive appositive.
    if first_sentence.lower().startswith(("rachel ", "margaret ")) and "," in first_sentence:
        first_sentence = first_sentence.split(",", 1)[0]
    if len(first_sentence) > 180:
        first_sentence = first_sentence[:177].rstrip() + "..."
    return first_sentence or fallback


def _scenario_presentation(session=None, prompt=None):
    role_text = _scenario_text_for_session(session) if session else getattr(prompt, "content", "")
    scenario = parse_scenario_prompt(role_text) if role_text else None
    fallback = getattr(prompt, "title", "") or getattr(getattr(session, "role_prompt", None), "title", "") or "Simulated character"
    if not scenario:
        return {
            "character_label": fallback,
            "character_summary": fallback,
            "learner_summary": "the learner in this simulation",
            "briefing_context": "",
        }
    background = re.split(r"(?<=[.!?])\s+|\n", scenario.background_context.strip(), maxsplit=1)[0].strip()
    if len(background) > 280:
        background = background[:277].rstrip() + "..."
    character = _role_summary(scenario.character, fallback)
    return {
        "character_label": character,
        "character_summary": character,
        "learner_summary": _role_summary(scenario.learner, "the learner in this simulation"),
        "briefing_context": background,
    }


def _legacy_introduction_message_id(session):
    """Identify introductions stored as assistant messages by older sessions."""
    if not session:
        return None
    candidate = (
        session.messages.filter(sender=ChatMessage.Sender.ASSISTANT, turn_id="")
        .order_by("created_at", "id")
        .first()
    )
    if candidate and candidate.content.strip() == _simulation_introduction(session=session):
        return candidate.id
    return None


def _conversation_messages(session):
    """Return only roleplay turns passed to ConversationEngine."""
    messages = session.messages.order_by("created_at", "id")
    legacy_intro_id = _legacy_introduction_message_id(session)
    return messages.exclude(pk=legacy_intro_id) if legacy_intro_id else messages


def _rendered_chat_messages(session, user_label):
    if not session:
        return []

    presentation = _scenario_presentation(session=session)
    legacy_intro_id = _legacy_introduction_message_id(session)
    rendered = []
    for message in session.messages.order_by("created_at", "id"):
        if message.id == legacy_intro_id:
            continue
        display_content = message.content
        embedded_emotion = ""
        if message.sender == ChatMessage.Sender.ASSISTANT:
            _legacy_dialogue, embedded_emotion = split_dialogue_and_voice(message.content)
            display_content = clean_dialogue_for_display(message.content)
        rendered.append(
            {
                "id": message.id,
                "sender": message.sender,
                "sender_display": presentation["character_label"] if message.sender == ChatMessage.Sender.ASSISTANT else user_label,
                "created_at": message.created_at,
                "display_content": display_content,
                "voice_enabled": bool(message.voice_metadata or embedded_emotion),
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
    latest = current_session.messages.order_by("-id").first() if current_session else None
    pending = latest if latest and latest.sender == ChatMessage.Sender.STUDENT else None
    voice_call = (
        current_session.voice_conversations.order_by("-created_at").first()
        if current_session and current_session.interaction_mode == ChatSession.InteractionMode.VOICE
        else None
    )
    presentation = _scenario_presentation(session=current_session, prompt=selected_prompt)
    scenario = parse_scenario_prompt(
        (current_session.scenario_content if current_session else "")
        or (selected_prompt.content if selected_prompt else "")
    )
    voice_configured = bool(scenario.introduction_voice_id and scenario.roleplay_voice_id)
    return {
        "active_prompts": active_prompts,
        "selected_prompt": selected_prompt,
        "sessions": sessions,
        "current_session": current_session,
        "voice_call_failed": bool(voice_call and voice_call.status == VoiceConversation.Status.ERROR),
        "rendered_messages": _rendered_chat_messages(current_session, user_label),
        "error_message": error_message,
        "force_new": force_new,
        "pending_message": pending,
        "simulation_introduction": _simulation_introduction(session=current_session, prompt=selected_prompt),
        "character_label": presentation["character_label"],
        "scenario_character": presentation["character_summary"],
        "scenario_learner": presentation["learner_summary"],
        "scenario_briefing_context": presentation["briefing_context"],
        "show_mode_selection": bool(selected_prompt and current_session is None),
        "is_text_session": bool(current_session and current_session.interaction_mode == ChatSession.InteractionMode.TEXT),
        "is_voice_session": bool(current_session and current_session.interaction_mode == ChatSession.InteractionMode.VOICE),
        # This is deliberately only a capability flag. The page never receives
        # the API key, Speech Engine ID, or a reusable provider credential.
        "speech_engine_enabled": is_speech_engine_configured(),
        "voice_enabled": is_speech_engine_configured() and voice_configured,
        "voice_configuration_ready": voice_configured,
    }


def _selected_chat_state(request, *, restore_latest_session=True):
    prompt_queryset = RolePrompt.objects.all() if _is_professor(request.user) else RolePrompt.objects.filter(is_active=True)
    prompt_candidates = _deactivate_incomplete_prompts(
        list(prompt_queryset.order_by("title"))
    )
    # Professors may test complete drafts, but an incomplete prompt is never
    # offered to either role. Existing saved sessions remain independently
    # viewable through their scenario snapshots.
    active_prompts = [
        prompt
        for prompt in prompt_candidates
        if prompt.is_complete and (_is_professor(request.user) or prompt.is_active)
    ]
    sessions = (
        ChatSession.objects.filter(student=request.user)
        .select_related("role_prompt")
        .prefetch_related("messages")
        .order_by("-started_at")
    )

    selected_prompt = None
    selected_prompt_id = request.POST.get("prompt_id") or request.GET.get("prompt")
    if selected_prompt_id:
        selected_prompt = next(
            (prompt for prompt in active_prompts if str(prompt.pk) == str(selected_prompt_id)),
            None,
        )
    elif active_prompts:
        selected_prompt = active_prompts[0]

    current_session = None
    session_id = request.POST.get("session_id") or request.GET.get("session")
    if session_id:
        current_session = get_object_or_404(
            ChatSession.objects.filter(student=request.user).select_related("role_prompt"),
            pk=session_id,
        )
        if (
            selected_prompt_id
            and str(current_session.role_prompt_id) == str(selected_prompt_id)
            and selected_prompt is None
        ):
            # An incomplete/deactivated prompt may still label a historical
            # transcript; it cannot be used to start another conversation.
            selected_prompt = current_session.role_prompt
        elif selected_prompt_id and current_session.role_prompt_id != getattr(selected_prompt, "id", None):
            current_session = (
                sessions.filter(role_prompt=selected_prompt).first()
                if restore_latest_session
                else None
            )
        elif current_session.role_prompt and not selected_prompt_id:
            selected_prompt = current_session.role_prompt
    elif restore_latest_session and request.GET.get("new") != "1" and selected_prompt:
        current_session = sessions.filter(role_prompt=selected_prompt).first()

    force_new = request.POST.get("force_new") == "1" or request.GET.get("force_new") == "1"
    return active_prompts, sessions, selected_prompt, current_session, force_new


def _usable_chat_session(
    user,
    selected_prompt,
    current_session,
    force_new,
    turn_id="",
    interaction_mode=ChatSession.InteractionMode.TEXT,
):
    needs_session = (
        not current_session
        or current_session.role_prompt_id != selected_prompt.id
        or current_session.interaction_mode != interaction_mode
        or (current_session.ended_at is not None and not turn_id)
        or force_new
    )
    if needs_session:
        current_session = None
        if not force_new:
            current_session = (
                ChatSession.objects.filter(
                    student=user,
                    role_prompt=selected_prompt,
                    interaction_mode=interaction_mode,
                    ended_at__isnull=True,
                )
                .order_by("-started_at")
                .first()
            )
        return current_session or _create_chat_session(user, selected_prompt, interaction_mode), None

    return current_session, None


def _create_chat_session(user, selected_prompt, interaction_mode=ChatSession.InteractionMode.TEXT):
    """Create the durable chat; the narrator introduction is presentation-only."""
    return ChatSession.objects.create(
        student=user,
        role_prompt=selected_prompt,
        scenario_content=selected_prompt.content,
        interaction_mode=interaction_mode,
    )


def _chat_dashboard(
    request,
    *,
    template_name,
    view_name,
    user_label,
    no_prompt_message,
    allow_delete_session=False,
    choose_mode=False,
    restore_latest_session=True,
):
    active_prompts, sessions, selected_prompt, current_session, force_new = _selected_chat_state(
        request,
        restore_latest_session=restore_latest_session,
    )
    error_message = None

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "new_session":
            if selected_prompt and not selected_prompt.is_complete:
                error_message = "Complete every required prompt field before testing this scenario."
            elif selected_prompt:
                if choose_mode:
                    return redirect(_chat_url(view_name, selected_prompt, new=1))
                session = _create_chat_session(request.user, selected_prompt, ChatSession.InteractionMode.TEXT)
                return redirect(_chat_url(view_name, selected_prompt, session=session.id))
            elif not error_message:
                return redirect(_chat_url(view_name, selected_prompt))

        if choose_mode and action == "start_session":
            mode = request.POST.get("mode", "").strip()
            if not selected_prompt:
                error_message = no_prompt_message
            elif not selected_prompt.is_complete:
                error_message = "Complete every required prompt field before testing this scenario."
            elif mode not in ChatSession.InteractionMode.values:
                error_message = "Choose voice or text practice to continue."
            elif mode == ChatSession.InteractionMode.VOICE and not is_speech_engine_configured():
                error_message = "Voice practice is not configured yet. Choose text practice or ask your instructor to enable it."
            elif mode == ChatSession.InteractionMode.VOICE:
                scenario = parse_scenario_prompt(selected_prompt.content)
                if not scenario.introduction_voice_id or not scenario.roleplay_voice_id:
                    error_message = "Choose both an introduction voice and a roleplay voice in the prompt editor before starting voice practice."
            if not error_message:
                session = _create_chat_session(request.user, selected_prompt, mode)
                return redirect(_chat_url(view_name, selected_prompt, session=session.id))

        if allow_delete_session and action == "delete_session":
            target_session_id = request.POST.get("target_session_id") or request.POST.get("session_id")
            if target_session_id:
                get_object_or_404(ChatSession, pk=target_session_id, student=request.user).delete()
            return redirect(_chat_url(view_name, selected_prompt, new=1))

        if action == "close_conversation":
            if current_session and current_session.ended_at is None:
                _release_learner_turn(
                    current_session,
                    current_session.active_turn_id,
                    current_session.active_claim_id,
                )
                current_session.ended_at = timezone.now()
                current_session.save(update_fields=["ended_at"])
            return redirect(_chat_url(view_name, selected_prompt, session=current_session.id) if current_session else _chat_url(view_name, selected_prompt))

        if action == "send_message":
            user_text = request.POST.get("message", "").strip()
            requested_turn_id = request.POST.get("turn_id", "").strip()
            turn_id = _request_turn_id(requested_turn_id)
            retry_turn_id = turn_id if requested_turn_id == turn_id else ""
            if not selected_prompt:
                error_message = no_prompt_message
            elif not current_session and not selected_prompt.is_complete:
                error_message = "Complete every required prompt field before testing this scenario."
            elif not user_text:
                error_message = "Please type a message before sending."
            else:
                current_session, error_message = _usable_chat_session(
                    request.user,
                    selected_prompt,
                    current_session,
                    force_new,
                    turn_id=retry_turn_id,
                    interaction_mode=ChatSession.InteractionMode.TEXT,
                )

                if not error_message:
                    completed_turns = current_session.messages.filter(
                        sender=ChatMessage.Sender.STUDENT
                    ).count()
                    retry_exists = bool(
                        retry_turn_id
                        and current_session.messages.filter(turn_id=retry_turn_id).exists()
                    )
                    if completed_turns >= MAX_TURNS and not retry_exists:
                        error_message = "This conversation reached its safety limit before a natural ending."

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

            current_session, turn_status, existing_assistant = _accept_learner_turn(
                current_session,
                user_text,
                turn_id,
            )
            if turn_status == "pending":
                error_message = "Please wait for the AI response before sending another message."
            elif turn_status == "closed":
                return redirect(_chat_url(view_name, selected_prompt, session=current_session.id))
            elif turn_status == "conflict":
                error_message = "This turn ID was already used for different message content. Please retry from the chat page."

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

            if turn_status == "duplicate":
                return redirect(_chat_url(view_name, selected_prompt, session=current_session.id))

            try:
                assistant_text, conversation_complete, debug_info = _generate_assistant_response(
                    current_session.scenario_content or selected_prompt.content,
                    _conversation_messages(current_session),
                    current_session,
                )
            except Exception:
                _release_learner_turn(current_session, turn_id, current_session.active_claim_id)
                logger.exception("Chat response generation failed session=%s turn=%s", current_session.id, turn_id)
                return render(request, template_name, _chat_dashboard_context(
                    active_prompts, selected_prompt, sessions, current_session,
                    "The response could not be generated. Your message is saved. Press Retry response; if this continues, contact your instructor.",
                    user_label, False,
                ), status=503)
            _persist_assistant_response(
                current_session,
                assistant_text,
                conversation_complete,
                debug_info,
                turn_id=turn_id,
                claim_id=current_session.active_claim_id,
            )
            logger.debug(
                "Chat response generated: user=%s session=%s phase=%s complete=%s reason=%s ending_check=%s",
                request.user.username,
                current_session.id,
                debug_info.get("phase"),
                conversation_complete,
                debug_info.get("reason"),
                debug_info.get("ending_check"),
            )
            return redirect(
                _chat_url(
                    view_name,
                    selected_prompt,
                    session=current_session.id,
                    autoplay=1,
                )
            )

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
        user_label="YOU",
        no_prompt_message="No scenario is available. Create or upload a scenario first.",
        allow_delete_session=True,
        choose_mode=True,
        restore_latest_session=False,
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
        user_label="YOU",
        no_prompt_message="No active prompt is available. Ask your professor to activate one.",
        allow_delete_session=True,
        choose_mode=True,
    )


def _voice_json_payload(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _voice_error(message, status=400):
    return JsonResponse({"error": message}, status=status)


def _finalize_voice_call(voice_call, *, failed=False):
    """Close the provider run and its durable transcript together.

    A provider close/error and the browser's End Call action can arrive in
    either order.  The first terminal state wins so an expected disconnect
    after End Call is never later displayed as a failed call.
    """
    if voice_call.ended_at is None:
        voice_call.status = VoiceConversation.Status.ERROR if failed else VoiceConversation.Status.ENDED
        voice_call.failure_reason = "Voice connection ended unexpectedly." if failed else ""
        voice_call.ended_at = voice_call.ended_at or timezone.now()
        voice_call.save(update_fields=["status", "ended_at", "failure_reason"])

    chat_session = voice_call.chat_session
    if chat_session.ended_at is None:
        _release_learner_turn(
            chat_session,
            chat_session.active_turn_id,
            chat_session.active_claim_id,
        )
        chat_session.ended_at = timezone.now()
        chat_session.save(update_fields=["ended_at"])


def _voice_session_for_request(user, prompt, payload):
    """Return the selected open voice session or create one for voice start."""
    requested_session_id = str(payload.get("session_id") or "").strip()
    force_new = bool(payload.get("force_new"))
    current_session = None
    if requested_session_id and requested_session_id.isdigit() and not force_new:
        current_session = ChatSession.objects.filter(student=user, pk=requested_session_id).first()
        if (
            current_session
            and (
                current_session.role_prompt_id != prompt.id
                or current_session.interaction_mode != ChatSession.InteractionMode.VOICE
                or current_session.ended_at is not None
            )
        ):
            current_session = None
    if current_session is None:
        current_session = _create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
    return current_session


@login_required
@require_POST
def student_voice_token(request):
    """Create a short-lived browser token without exposing provider secrets."""
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")

    prompt_id = str(payload.get("prompt_id") or "").strip()
    prompts = RolePrompt.objects.all() if _is_professor(request.user) else RolePrompt.objects.filter(is_active=True)
    prompt = prompts.filter(pk=prompt_id).first()
    if not prompt:
        return _voice_error("Choose an available scenario before starting voice practice.", status=404)
    if not prompt.is_complete:
        if prompt.is_active:
            prompt.is_active = False
            prompt.save(update_fields=["is_active", "updated_at"])
        return _voice_error(
            "Complete every required prompt field before testing this scenario.",
            status=409,
        )
    if not is_speech_engine_configured():
        return _voice_error("Voice practice is not configured yet. Ask your instructor to enable it.", status=503)

    chat_session = _voice_session_for_request(request.user, prompt, payload)
    scenario = parse_scenario_prompt(chat_session.scenario_content or prompt.content)
    if not scenario.introduction_voice_id or not scenario.roleplay_voice_id:
        return _voice_error(
            "This scenario needs both an introduction voice and a roleplay voice before voice practice can start.",
            status=409,
        )
    if chat_session.active_turn_id:
        return _voice_error("The current chat is still preparing a response. Please wait before starting voice practice.", status=409)

    voice_call = VoiceConversation.objects.create(chat_session=chat_session)
    try:
        token = issue_webrtc_token(
            participant_name=request.user.username,
            voice_id=scenario.roleplay_voice_id,
        )
    except SpeechEngineConfigurationError as error:
        voice_call.status = VoiceConversation.Status.ERROR
        voice_call.failure_reason = "Voice service is not configured."
        voice_call.save(update_fields=["status", "failure_reason"])
        logger.warning("Speech Engine token configuration failed: %s", error)
        return _voice_error("Voice practice is not configured yet. Ask your instructor to enable it.", status=503)
    except Exception:
        voice_call.status = VoiceConversation.Status.ERROR
        voice_call.failure_reason = "Could not issue a voice session token."
        voice_call.save(update_fields=["status", "failure_reason"])
        logger.exception("Speech Engine token request failed user=%s", request.user.pk)
        return _voice_error("Voice service is temporarily unavailable. Please try again.", status=503)

    if token.provider_conversation_id:
        try:
            with transaction.atomic():
                voice_call = VoiceConversation.objects.select_for_update().get(pk=voice_call.pk)
                existing = VoiceConversation.objects.filter(
                    provider_conversation_id=token.provider_conversation_id
                ).exclude(pk=voice_call.pk).exists()
                if existing:
                    raise ValueError("Provider conversation ID was already mapped.")
                voice_call.provider_conversation_id = token.provider_conversation_id
                voice_call.save(update_fields=["provider_conversation_id"])
        except Exception:
            voice_call.status = VoiceConversation.Status.ERROR
            voice_call.failure_reason = "Could not map the voice conversation."
            voice_call.save(update_fields=["status", "failure_reason"])
            logger.exception("Speech Engine token mapping failed voice_call=%s", voice_call.pk)
            return _voice_error("Voice service could not start. Please try again.", status=503)

    presentation = _scenario_presentation(session=chat_session)
    response = JsonResponse(
        {
            "token": token.token,
            "voice_call_id": str(voice_call.pk),
            "chat_session_id": chat_session.id,
            "introduction": _simulation_introduction(session=chat_session),
            "introduction_audio_url": reverse("vip:student_voice_introduction", args=[chat_session.id]),
            "download_url": reverse("vip:student_download_session", args=[chat_session.id]),
            "character_label": presentation["character_label"],
            "max_duration_seconds": speech_engine_max_duration_seconds(),
        }
    )
    response["Cache-Control"] = "no-store"
    return response


@login_required
@require_POST
def student_voice_bind(request):
    """Fallback mapping for SDK/API versions that reveal the ID after connect."""
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    provider_conversation_id = str(payload.get("provider_conversation_id") or "").strip()
    if not call_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", provider_conversation_id):
        return _voice_error("Invalid voice conversation identifier.")

    try:
        with transaction.atomic():
            voice_call = VoiceConversation.objects.select_for_update().filter(
                pk=call_id,
                chat_session__student=request.user,
            ).first()
            if not voice_call:
                return _voice_error("Voice session was not found.", status=404)
            if voice_call.provider_conversation_id and voice_call.provider_conversation_id != provider_conversation_id:
                return _voice_error("Voice session cannot be rebound to a different conversation.", status=409)
            existing = VoiceConversation.objects.filter(
                provider_conversation_id=provider_conversation_id
            ).exclude(pk=voice_call.pk).exists()
            if existing:
                return _voice_error("Voice conversation is already mapped.", status=409)
            if not voice_call.provider_conversation_id:
                voice_call.provider_conversation_id = provider_conversation_id
                voice_call.save(update_fields=["provider_conversation_id"])
    except (ValueError, ValidationError):
        return _voice_error("Voice session was not found.", status=404)
    return JsonResponse({"ok": True})


@login_required
@require_POST
def student_voice_ready(request):
    """Open Speech Engine input after the local narrator has fully finished.

    The browser does not create the live Speech Engine session until narrator
    playback has ended. This marker remains a second, durable boundary: the
    adapter rejects every transcript until this endpoint succeeds.
    """
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    try:
        with transaction.atomic():
            voice_call = VoiceConversation.objects.select_for_update().filter(
                pk=call_id,
                chat_session__student=request.user,
            ).first()
            if not voice_call:
                return _voice_error("Voice session was not found.", status=404)
            if voice_call.status in {VoiceConversation.Status.ENDED, VoiceConversation.Status.ERROR}:
                return _voice_error("Voice session has already ended.", status=409)
            if voice_call.introduction_completed_at is None:
                voice_call.introduction_completed_at = timezone.now()
                voice_call.save(update_fields=["introduction_completed_at"])
    except (ValueError, ValidationError):
        return _voice_error("Voice session was not found.", status=404)
    return JsonResponse({"ok": True, "introduction_complete": True})


@login_required
@require_POST
def student_voice_end(request):
    """Finalize a voice run and close its saved transcript."""
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    failed = bool(payload.get("failed"))
    try:
        with transaction.atomic():
            voice_call = VoiceConversation.objects.select_for_update().select_related("chat_session").filter(
                pk=call_id,
                chat_session__student=request.user,
            ).first()
            if not voice_call:
                return _voice_error("Voice session was not found.", status=404)
            _finalize_voice_call(voice_call, failed=failed)
    except (ValueError, ValidationError):
        return _voice_error("Voice session was not found.", status=404)
    return JsonResponse({"ok": True})


@login_required
@require_POST
def student_voice_completion(request):
    """Report whether the last persisted roleplay response completed the chat.

    The browser asks after receiving an agent message, then waits for
    ElevenLabs' post-playback completion event before ending the provider
    connection.  No user input or ConversationEngine turn is created here.
    """
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    try:
        voice_call = VoiceConversation.objects.select_related("chat_session").filter(
            pk=call_id,
            chat_session__student=request.user,
        ).first()
    except (ValueError, ValidationError):
        voice_call = None
    if not voice_call:
        return _voice_error("Voice session was not found.", status=404)
    return JsonResponse(
        {
            "ok": True,
            "conversation_complete": bool(voice_call.chat_session.completion_status),
            "session_ended": voice_call.chat_session.ended_at is not None,
        }
    )


@login_required
@require_POST
def student_voice_mute(request):
    """Persist the microphone gate without changing conversation state."""
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    muted = bool(payload.get("muted"))
    try:
        voice_call = VoiceConversation.objects.filter(
            pk=call_id,
            chat_session__student=request.user,
        ).first()
    except (ValueError, ValidationError):
        voice_call = None
    if not voice_call:
        return _voice_error("Voice session was not found.", status=404)
    if voice_call.status in {VoiceConversation.Status.ENDED, VoiceConversation.Status.ERROR}:
        return _voice_error("Voice session has already ended.", status=409)
    voice_call.is_muted = muted
    voice_call.save(update_fields=["is_muted"])
    return JsonResponse({"ok": True, "muted": muted})


@login_required
@require_POST
def student_voice_connection_diagnostic(request):
    """Record sanitized browser-side SpeechEngine disconnect context."""
    if not _can_use_voice_practice(request.user):
        return _voice_error("Voice practice is available to student and professor accounts.", status=403)
    payload = _voice_json_payload(request)
    if payload is None:
        return _voice_error("Invalid voice request.")
    call_id = str(payload.get("voice_call_id") or "").strip()
    try:
        voice_call = VoiceConversation.objects.select_related("chat_session").filter(
            pk=call_id,
            chat_session__student=request.user,
        ).first()
    except (ValueError, ValidationError):
        voice_call = None
    if not voice_call:
        return _voice_error("Voice session was not found.", status=404)

    event = str(payload.get("event") or "unknown").strip().lower()
    if not re.fullmatch(r"[a-z_-]{1,32}", event):
        event = "unknown"
    connection_state = str(payload.get("connection_state") or "").strip().lower()
    connection_state = re.sub(r"[^a-z0-9_-]", "", connection_state)[:64]
    close_code = payload.get("close_code")
    if not isinstance(close_code, int) or isinstance(close_code, bool) or not 0 <= close_code <= 65535:
        close_code = None
    close_reason = re.sub(r"\s+", " ", str(payload.get("close_reason") or "")).strip()[:240]
    browser_provider_id = str(payload.get("provider_conversation_id") or "").strip()[:128]
    logger.warning(
        "SpeechEngine browser connection event=%s state=%s close_code=%s close_reason=%r "
        "provider_conversation=%s browser_provider_conversation=%s voice_call=%s chat_session=%s",
        event,
        connection_state,
        close_code,
        close_reason,
        voice_call.provider_conversation_id,
        browser_provider_id,
        voice_call.pk,
        voice_call.chat_session_id,
    )
    return JsonResponse({"ok": True})


@login_required
def student_voice_introduction(request, session_id):
    """Stream the presentation-only introduction in a neutral narrator voice."""
    if not _can_use_voice_practice(request.user):
        return redirect("vip:home")
    session = get_object_or_404(
        ChatSession.objects.select_related("role_prompt"),
        pk=session_id,
        student=request.user,
        interaction_mode=ChatSession.InteractionMode.VOICE,
    )
    introduction = _simulation_introduction(session=session)
    scenario = parse_scenario_prompt(session.scenario_content or getattr(session.role_prompt, "content", ""))
    provider_api_key = elevenlabs_api_key()
    if not provider_api_key or not scenario.introduction_voice_id:
        return HttpResponseBadRequest("Narrator audio is not configured.")

    try:
        stream = ElevenLabsSpeechStream(
            provider_api_key,
            voice_id=scenario.introduction_voice_id,
            text=introduction,
        )
        result = StreamingHttpResponse(stream, content_type="audio/mpeg")
        result["Cache-Control"] = "no-store"
        result["X-Accel-Buffering"] = "no"
        return result
    except Exception:
        logger.exception("Narrator speech generation failed session=%s", session.id)
        return HttpResponse("Narrator audio is temporarily unavailable. Please try again.", status=503)


@login_required
def student_download_session(request, session_id):
    if not (_is_student(request.user) or _is_professor(request.user)):
        return redirect("vip:home")

    session = get_object_or_404(
        ChatSession.objects.select_related("student", "role_prompt"),
        pk=session_id,
        student=request.user,
    )
    messages = _conversation_messages(session)

    lines = [
        f"Student: {session.student.username}",
        f"Prompt: {session.role_prompt.title if session.role_prompt else 'None'}",
        f"Session ID: {session.id}",
        f"Mode: {session.get_interaction_mode_display()}",
        f"Started: {session.started_at}",
        "",
        f"SIMULATION: {_simulation_introduction(session=session)}",
        "",
    ]
    for message in messages:
        speaker = "YOU" if message.sender == ChatMessage.Sender.STUDENT else _scenario_presentation(session=session)["character_label"].upper()
        content = message.content
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

    dialogue = clean_dialogue_for_display(message.content)
    _legacy_dialogue, embedded_emotion = split_dialogue_and_voice(message.content)
    emotion = getattr(message, "voice_metadata", "") or embedded_emotion
    use_emotion_voice = request.GET.get("emotion", "1") != "0"
    if not emotion:
        return HttpResponseBadRequest("Audio is not available for this text-only assistant message.")

    api_key = _load_api_key()
    if not api_key:
        return HttpResponseBadRequest("API key is not configured.")

    try:
        from .speech import SpeechStream

        stream = SpeechStream(
            api_key,
            model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice="coral",
            input=dialogue or "No spoken dialogue.",
            instructions=emotion if use_emotion_voice else "Speak naturally and clearly.",
            response_format="mp3",
        )
        result = StreamingHttpResponse(stream, content_type="audio/mpeg")
        result["Cache-Control"] = "no-store"
        result["X-Accel-Buffering"] = "no"
        return result
    except Exception:
        logger.exception("Speech generation failed message=%s", message.id)
        return HttpResponse("Voice is temporarily unavailable. You can continue using text.", status=503)
