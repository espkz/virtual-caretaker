import os
import re
import io
import csv
import json
import logging
import queue
import threading
import uuid
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
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
from .models import ChatMessage, ChatSession, RolePrompt
from .prompt_utils import find_section_by_aliases, split_markdown_sections
from .conversation_graph import MAX_TURNS
from .conversation_engine import (
    DEFAULT_INTRODUCTION,
    ConversationEngine,
    _repeated_answered_question,
    split_dialogue_and_voice,
)
from .conversation_scenario import parse_scenario_prompt
from .voice_timing import VoicePipelineTiming

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
    return {
        "role": role or role_text.strip(),
        "learner_role": learner_role or "nursing student",
        "voice_gender": voice_gender,
        "voice_style": voice_style.strip(),
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


def _generate_assistant_response(
    role_text,
    session_messages,
    session=None,
    stream_callback=None,
    timing_callback=None,
):
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
    return engine.respond(
        role_text,
        list(session_messages),
        conversation_state=conversation_state,
        stream_callback=stream_callback,
        timing_callback=timing_callback,
    )


def _wants_voice_stream(request):
    return (
        request.method == "POST"
        and request.POST.get("action") == "send_message"
        and "text/event-stream" in request.headers.get("Accept", "")
    )


def _sse(event_name, payload=None):
    data = {"event": event_name, **(payload or {})}
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_SENTENCE_BOUNDARY_RE = re.compile(r"^(.+?[.!?](?:[\"'’\)\]]*)?(?:\s+|$))", flags=re.DOTALL)


def _take_completed_speech_chunks(buffer, flush=False):
    """Yield complete sentence-sized chunks and retain an incomplete suffix."""
    chunks = []
    remaining = buffer or ""
    while remaining:
        match = _SENTENCE_BOUNDARY_RE.match(remaining)
        if not match:
            break
        chunks.append(match.group(1).strip())
        remaining = remaining[match.end():]
    if flush and remaining.strip():
        chunks.append(remaining.strip())
        remaining = ""
    return chunks, remaining


def _request_turn_id(value):
    """Accept a safe client retry ID or create one for a non-voice submit."""
    candidate = (value or "").strip()
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate):
        return candidate
    return uuid.uuid4().hex


def _accept_learner_turn(session, user_text, turn_id):
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
            locked_session.save(update_fields=["active_turn_id", "active_claim_id"])
            return locked_session, "accepted", None

        if locked_session.active_turn_id:
            return locked_session, "pending", None

        latest = locked_session.messages.order_by("-id").first()
        if latest and latest.sender == ChatMessage.Sender.STUDENT:
            return locked_session, "pending", None

        ChatMessage.objects.create(
            session=locked_session,
            sender=ChatMessage.Sender.STUDENT,
            content=user_text,
            turn_id=turn_id,
        )
        locked_session.active_turn_id = turn_id
        locked_session.active_claim_id = uuid.uuid4().hex
        locked_session.save(update_fields=["active_turn_id", "active_claim_id"])
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
        locked_session.save(update_fields=["active_turn_id", "active_claim_id"])
    return True


def _merge_pipeline_timing(existing, incoming):
    """Merge a later client/TTS report into the stored turn trace.

    A turn can produce several TTS requests (one per sentence), so timing
    reports arrive at different times.  First marks are retained while new
    event names and derived durations are added.
    """
    existing = existing if isinstance(existing, dict) else {}
    incoming = incoming if isinstance(incoming, dict) else {}
    client_events = dict(existing.get("client_events") or {})
    client_events.update(incoming.get("client_events") or {})
    server_events = dict(existing.get("server_events") or {})
    for name, value in (incoming.get("server_events") or {}).items():
        server_events.setdefault(name, value)

    timing = VoicePipelineTiming(
        {
            "trace_id": incoming.get("trace_id") or existing.get("trace_id"),
            "events": client_events,
            "server_events": server_events,
        }
    )
    timing.events.update(server_events)
    timing.client_events.update(client_events)
    return timing.snapshot()


def _save_message_timing(message_id, timing_snapshot):
    """Merge a timing snapshot onto one accessible transcript message."""
    if not message_id or not isinstance(timing_snapshot, dict):
        return False
    with transaction.atomic():
        message = ChatMessage.objects.select_for_update().get(pk=message_id)
        message.pipeline_timing = _merge_pipeline_timing(message.pipeline_timing, timing_snapshot)
        message.save(update_fields=["pipeline_timing"])
    return True


def _save_turn_timing(session, turn_id, timing_snapshot):
    """Attach a turn trace to its assistant row, or its learner row if empty."""
    if not session or not turn_id or not isinstance(timing_snapshot, dict):
        return False
    with transaction.atomic():
        locked_session = ChatSession.objects.select_for_update().get(pk=session.pk)
        message = locked_session.messages.select_for_update().filter(
            sender=ChatMessage.Sender.ASSISTANT,
            turn_id=turn_id,
        ).first()
        if message is None:
            message = locked_session.messages.select_for_update().filter(
                sender=ChatMessage.Sender.STUDENT,
                turn_id=turn_id,
            ).first()
        if message is None:
            return False
        message.pipeline_timing = _merge_pipeline_timing(message.pipeline_timing, timing_snapshot)
        message.save(update_fields=["pipeline_timing"])
    return True


def _persist_assistant_response(
    session,
    assistant_text,
    conversation_complete,
    debug_info,
    turn_id="",
    cancellation_callback=None,
    claim_id="",
    timing_snapshot=None,
):
    """Commit at most one response, and never commit for a stale turn."""
    with transaction.atomic():
        locked_session = ChatSession.objects.select_for_update().get(pk=session.pk)
        if turn_id:
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
                pipeline_timing=timing_snapshot if isinstance(timing_snapshot, dict) else {},
            )
        locked_session.conversation_stage = debug_info.get("current_stage", locked_session.conversation_stage)
        locked_session.conversation_phase = debug_info.get("phase", locked_session.conversation_phase)
        locked_session.completion_status = conversation_complete
        update_fields = ["conversation_stage", "conversation_phase", "completion_status"]
        if turn_id:
            locked_session.active_turn_id = ""
            locked_session.active_claim_id = ""
            locked_session.last_completed_turn_id = turn_id
            update_fields.extend(["active_turn_id", "active_claim_id", "last_completed_turn_id"])
        if conversation_complete and locked_session.ended_at is None:
            locked_session.ended_at = timezone.now()
            update_fields.append("ended_at")
        locked_session.save(update_fields=update_fields)
    return True


def _stream_chat_response(request, role_text, session, selected_prompt, view_name, user_label, turn_id=""):
    """Stream one accepted turn while preserving the normal persistence path."""
    timing = VoicePipelineTiming(request.POST.get("voice_timing"))
    scenario = parse_scenario_prompt(role_text)
    provisional_voice = scenario.voice_metadata()
    events = queue.Queue()
    streamed_text = ""
    speech_index = 0
    stream_voice = provisional_voice
    event_sequence = 0
    cancellation_event = threading.Event()
    claim_id = getattr(session, "active_claim_id", "")

    def emit(event_name, payload=None):
        nonlocal event_sequence
        event_sequence += 1
        return _sse(
            event_name,
            {
                "turn_id": turn_id,
                "session_id": session.id,
                "event_sequence": event_sequence,
                **(payload or {}),
            },
        )

    def timing_callback(name):
        timing.mark(name)
        events.put({"kind": "timing", "name": name})

    def stream_callback(event):
        events.put(event)

    def worker():
        try:
            result = _generate_assistant_response(
                role_text,
                session.messages.order_by("created_at"),
                session,
                stream_callback=stream_callback,
                timing_callback=timing_callback,
            )
            events.put({"kind": "result", "result": result})
        except Exception as exc:
            logger.exception("Streaming voice response failed trace=%s", timing.trace_id)
            _release_learner_turn(session, turn_id, claim_id)
            events.put({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put({"kind": "worker_done"})

    def generate():
        nonlocal streamed_text, speech_index, stream_voice
        try:
            yield emit(
                "stream_started",
                {
                    "trace_id": timing.trace_id,
                    "voice_metadata": provisional_voice,
                    "timings": timing.snapshot(),
                },
            )
            thread = threading.Thread(target=worker, name=f"voice-response-{timing.trace_id}", daemon=True)
            thread.start()
            result = None
            worker_error = None
            while result is None and worker_error is None:
                item = events.get()
                kind = item.get("kind")
                if kind == "gpt_delta":
                    delta = item.get("text", "")
                    if not delta:
                        continue
                    streamed_text += delta
                    yield emit("assistant_delta", {"text": delta, "trace_id": timing.trace_id})
                elif kind == "voice_metadata":
                    stream_voice = item.get("value") or stream_voice
                    yield emit(
                        "voice_metadata",
                        {"value": stream_voice, "trace_id": timing.trace_id},
                    )
                elif kind == "stream_reset":
                    streamed_text = ""
                    yield emit("assistant_reset", {"trace_id": timing.trace_id})
                elif kind == "timing":
                    yield emit("timing", {"name": item.get("name"), "timings": timing.snapshot()})
                elif kind == "result":
                    result = item.get("result")
                elif kind == "error":
                    worker_error = item.get("error")
                elif kind == "worker_done" and result is not None:
                    break

            if worker_error:
                timing.log("error")
                yield emit("error", {"message": "The assistant response could not be generated."})
                return

            if cancellation_event.is_set():
                _release_learner_turn(session, turn_id, claim_id)
                timing.log("cancelled")
                return

            assistant_text, conversation_complete, debug_info = result
            if assistant_text != streamed_text:
                streamed_text = assistant_text or ""
                yield emit(
                    "assistant_final",
                    {"text": streamed_text, "trace_id": timing.trace_id},
                )
            elif not streamed_text and assistant_text:
                streamed_text = assistant_text
                yield emit("assistant_final", {"text": streamed_text, "trace_id": timing.trace_id})

            # Only the authoritative, post-repair response is eligible for TTS.
            # The previous implementation emitted sentence audio while a draft
            # was still provisional, so a repair could not retract that audio.
            persisted = _persist_assistant_response(
                session,
                assistant_text,
                conversation_complete,
                debug_info,
                turn_id=turn_id,
                cancellation_callback=cancellation_event.is_set,
                claim_id=claim_id,
                timing_snapshot=timing.snapshot(),
            )
            if not persisted:
                timing.log("stale")
                if not cancellation_event.is_set():
                    yield emit("error", {"message": "This response is no longer current."})
                return

            assistant_message = session.messages.filter(
                sender=ChatMessage.Sender.ASSISTANT,
                turn_id=turn_id,
            ).first()
            assistant_message_id = assistant_message.id if assistant_message else None
            chunks, _ = _take_completed_speech_chunks(assistant_text or "", flush=True)
            for chunk in chunks:
                speech_index += 1
                if speech_index == 1:
                    timing.mark("tts_text_first_usable")
                yield emit(
                    "speech_chunk",
                    {
                        "index": speech_index,
                        "text": chunk,
                        "voice_metadata": debug_info.get("voice_metadata") or stream_voice,
                        "trace_id": timing.trace_id,
                        "message_id": assistant_message_id,
                    },
                )

            timing.mark("response_generation_complete")
            timing.mark("response_complete")
            _save_turn_timing(session, turn_id, timing.snapshot())
            timing.log("complete")
            redirect_url = _chat_url(view_name, selected_prompt, session=session.id)
            yield emit(
                "complete",
                {
                    "trace_id": timing.trace_id,
                    "conversation_complete": conversation_complete,
                    "redirect_url": redirect_url,
                    "timings": timing.snapshot(),
                    "voice_metadata": debug_info.get("voice_metadata") or stream_voice,
                    "message_id": assistant_message_id,
                },
                )
        finally:
            cancellation_event.set()
            _release_learner_turn(session, turn_id, claim_id)

    return generate()


def _stream_committed_response(request, session, selected_prompt, view_name, assistant_message, turn_id):
    """Return an idempotent SSE replay for a completed client retry."""
    timing = VoicePipelineTiming(request.POST.get("voice_timing"))
    sequence = 0

    def emit(event_name, payload=None):
        nonlocal sequence
        sequence += 1
        return _sse(
            event_name,
            {
                "turn_id": turn_id,
                "session_id": session.id,
                "event_sequence": sequence,
                "trace_id": timing.trace_id,
                **(payload or {}),
            },
        )

    def generate():
        scenario = parse_scenario_prompt(selected_prompt.content)
        yield emit("stream_started", {"voice_metadata": scenario.voice_metadata()})
        if assistant_message:
            yield emit("assistant_final", {"text": assistant_message.content})
            chunks, _ = _take_completed_speech_chunks(assistant_message.content, flush=True)
            for index, chunk in enumerate(chunks, start=1):
                if index == 1:
                    timing.mark("tts_text_first_usable")
                yield emit(
                    "speech_chunk",
                    {
                        "index": index,
                        "text": chunk,
                        "voice_metadata": assistant_message.voice_metadata or scenario.voice_metadata(),
                        "message_id": assistant_message.id,
                    },
                )
        timing.mark("response_complete")
        yield emit(
            "complete",
            {
                "conversation_complete": session.completion_status,
                "redirect_url": _chat_url(view_name, selected_prompt, session=session.id),
                "timings": timing.snapshot(),
                "message_id": assistant_message.id if assistant_message else None,
                "voice_metadata": (
                    assistant_message.voice_metadata if assistant_message else scenario.voice_metadata()
                ),
            },
        )

    from django.http import StreamingHttpResponse

    response = StreamingHttpResponse(generate(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"
    return response


def _streaming_response(request, role_text, session, selected_prompt, view_name, user_label, turn_id=""):
    from django.http import StreamingHttpResponse

    response = StreamingHttpResponse(
        _stream_chat_response(request, role_text, session, selected_prompt, view_name, user_label, turn_id=turn_id),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache, no-transform"
    response["X-Accel-Buffering"] = "no"
    return response


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
        embedded_emotion = ""
        if message.sender == ChatMessage.Sender.ASSISTANT:
            display_content, embedded_emotion = _parse_dialogue_and_emotion(message.content)
        rendered.append(
            {
                "id": message.id,
                "sender": message.sender,
                "sender_display": "AI" if message.sender == ChatMessage.Sender.ASSISTANT else user_label,
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


def _usable_chat_session(user, selected_prompt, current_session, force_new, turn_id=""):
    needs_session = (
        not current_session
        or current_session.role_prompt_id != selected_prompt.id
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
                    ended_at__isnull=True,
                )
                .order_by("-started_at")
                .first()
            )
        return current_session or _create_chat_session(user, selected_prompt), None

    return current_session, None


def _seed_introduction(session):
    """Persist the fixed Introduction as text when a new chat starts."""
    if not session or not session.role_prompt:
        return
    if session.messages.filter(sender=ChatMessage.Sender.ASSISTANT).exists():
        return

    scenario = parse_scenario_prompt(session.role_prompt.content)
    introduction = (scenario.introduction or DEFAULT_INTRODUCTION).strip()
    if introduction:
        ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content=introduction,
        )


def _create_chat_session(user, selected_prompt):
    session = ChatSession.objects.create(student=user, role_prompt=selected_prompt)
    _seed_introduction(session)
    return session


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
            if selected_prompt:
                session = _create_chat_session(request.user, selected_prompt)
                return redirect(_chat_url(view_name, selected_prompt, session=session.id))
            return redirect(_chat_url(view_name, selected_prompt))

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
            return redirect(_chat_url(view_name, selected_prompt, new=1))

        if action == "send_message":
            user_text = request.POST.get("message", "").strip()
            requested_turn_id = request.POST.get("turn_id", "").strip()
            turn_id = _request_turn_id(requested_turn_id)
            retry_turn_id = turn_id if requested_turn_id == turn_id else ""
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
                    turn_id=retry_turn_id,
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

            _seed_introduction(current_session)
            current_session, turn_status, existing_assistant = _accept_learner_turn(
                current_session,
                user_text,
                turn_id,
            )
            if turn_status == "pending":
                error_message = "Please wait for the AI response before sending another message."
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
                if _wants_voice_stream(request):
                    return _stream_committed_response(
                        request,
                        current_session,
                        selected_prompt,
                        view_name,
                        existing_assistant,
                        turn_id,
                    )
                return redirect(_chat_url(view_name, selected_prompt, session=current_session.id))

            if _wants_voice_stream(request):
                return _streaming_response(
                    request,
                    selected_prompt.content,
                    current_session,
                    selected_prompt,
                    view_name,
                    user_label,
                    turn_id=turn_id,
                )

            normal_timing = VoicePipelineTiming(request.POST.get("voice_timing"))
            try:
                assistant_text, conversation_complete, debug_info = _generate_assistant_response(
                    selected_prompt.content,
                    current_session.messages.order_by("created_at"),
                    current_session,
                    timing_callback=normal_timing.mark,
                )
            except Exception:
                _release_learner_turn(current_session, turn_id, current_session.active_claim_id)
                logger.exception("Chat response generation failed session=%s turn=%s", current_session.id, turn_id)
                raise
            _persist_assistant_response(
                current_session,
                assistant_text,
                conversation_complete,
                debug_info,
                turn_id=turn_id,
                claim_id=current_session.active_claim_id,
                timing_snapshot=normal_timing.snapshot(),
            )
            normal_timing.mark("response_generation_complete")
            normal_timing.mark("response_complete")
            _save_turn_timing(current_session, turn_id, normal_timing.snapshot())
            normal_timing.log("complete")
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
                    trace=normal_timing.trace_id,
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
        voice_metadata = ""
        if message.sender == ChatMessage.Sender.ASSISTANT:
            content, embedded_voice = split_dialogue_and_voice(content)
            voice_metadata = (message.voice_metadata or embedded_voice).strip()
        lines.append(f"[{message.created_at}] {speaker}: {content}")
        if voice_metadata:
            lines.append(f"Voice metadata: {voice_metadata}")
        timing_snapshot = message.pipeline_timing or {}
        if timing_snapshot:
            lines.append("Voice pipeline timing:")
            lines.append(f"Trace ID: {timing_snapshot.get('trace_id', '')}")
            for section in ("client_events", "server_events"):
                events = timing_snapshot.get(section) or {}
                if events:
                    lines.append(f"{section.replace('_', ' ').title()}:")
                    for name, value in events.items():
                        lines.append(f"  {name}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
            durations = timing_snapshot.get("durations_ms") or {}
            if durations:
                lines.append(f"Durations (ms): {json.dumps(durations, ensure_ascii=False, sort_keys=True)}")
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

    dialogue, embedded_emotion = split_dialogue_and_voice(message.content)
    emotion = getattr(message, "voice_metadata", "") or embedded_emotion
    use_emotion_voice = request.GET.get("emotion", "1") != "0"
    if use_emotion_voice and not emotion:
        return HttpResponseBadRequest("Audio is not available for this text-only assistant message.")

    api_key = _load_api_key_from_txt()
    if not api_key:
        return HttpResponseBadRequest("API key is not configured.")

    timing = VoicePipelineTiming(trace_id=request.GET.get("trace"))
    timing.mark("tts_text_first_usable")
    timing.mark("tts_request_start")

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
            audio = response.read()
            timing.mark("tts_first_audio_data")
            timing.mark("tts_first_audio_chunk")
            timing.mark("tts_completion")
            timing.log("complete")
            _save_message_timing(message.id, timing.snapshot())
            stream = (audio[offset : offset + 64 * 1024] for offset in range(0, len(audio), 64 * 1024))
            result = StreamingHttpResponse(stream, content_type="audio/mpeg")
            result["Cache-Control"] = "no-store"
            result["X-Accel-Buffering"] = "no"
            return result

        from gtts import gTTS

        clean_text = _clean_for_tts(dialogue)
        audio_buffer = io.BytesIO()
        tts = gTTS(text=clean_text or "No content", lang="en")
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio = audio_buffer.read()
        timing.mark("tts_first_audio_data")
        timing.mark("tts_first_audio_chunk")
        timing.mark("tts_completion")
        timing.log("complete")
        _save_message_timing(message.id, timing.snapshot())
        stream = (audio[offset : offset + 64 * 1024] for offset in range(0, len(audio), 64 * 1024))
        result = StreamingHttpResponse(stream, content_type="audio/mpeg")
        result["Cache-Control"] = "no-store"
        result["X-Accel-Buffering"] = "no"
        return result
    except Exception as exc:
        timing.log("error")
        return HttpResponseBadRequest(f"TTS error: {exc}")


def _timing_message_for_params(request, params):
    """Resolve the transcript row allowed to receive a TTS timing trace."""
    message_id = params.get("message_id")
    if message_id:
        filters = {
            "pk": message_id,
            "sender": ChatMessage.Sender.ASSISTANT,
            "session__student": request.user,
        }
        if params.get("session_id"):
            filters["session_id"] = params.get("session_id")
        return get_object_or_404(ChatMessage.objects.select_related("session"), **filters)

    session_id = params.get("session_id")
    turn_id = params.get("turn_id")
    if session_id and turn_id:
        message = ChatMessage.objects.filter(
            session_id=session_id,
            session__student=request.user,
            sender=ChatMessage.Sender.ASSISTANT,
            turn_id=turn_id,
        ).first()
        if message is None:
            message = ChatMessage.objects.filter(
                session_id=session_id,
                session__student=request.user,
                sender=ChatMessage.Sender.STUDENT,
                turn_id=turn_id,
            ).first()
        return message
    return None


@login_required
def student_stream_tts(request):
    """Stream a short GPT-TTS sentence used by the incremental voice path."""
    if not (_is_student(request.user) or _is_professor(request.user)):
        return redirect("vip:home")
    if request.method not in {"GET", "POST"}:
        return HttpResponseBadRequest("Only GET and POST are supported.")

    params = request.GET if request.method == "GET" else request.POST
    dialogue = (params.get("text") or "").strip()
    if not dialogue:
        return HttpResponseBadRequest("TTS text is required.")
    if len(dialogue) > 4096:
        return HttpResponseBadRequest("TTS text is too long.")

    api_key = _load_api_key_from_txt()
    if not api_key:
        return HttpResponseBadRequest("API key is not configured.")

    emotion = (params.get("voice_metadata") or "Speak naturally and clearly.").strip()[:2000]
    use_emotion_voice = params.get("emotion", "1") != "0"
    timing = VoicePipelineTiming(trace_id=params.get("trace_id"))
    timing.mark("tts_text_first_usable")
    timing.mark("tts_request_start")
    timing_message = _timing_message_for_params(request, params)

    if not use_emotion_voice:
        try:
            from gtts import gTTS

            audio_buffer = io.BytesIO()
            gTTS(text=_clean_for_tts(dialogue) or "No content", lang="en").write_to_fp(audio_buffer)
            audio = audio_buffer.getvalue()
        except Exception as exc:
            timing.log("error")
            return HttpResponseBadRequest(f"TTS error: {exc}")

        def no_emotion_audio_iterator():
            failed = False
            try:
                for offset in range(0, len(audio), 64 * 1024):
                    chunk = audio[offset : offset + 64 * 1024]
                    if chunk:
                        if "tts_first_audio_data" not in timing.events:
                            timing.mark("tts_first_audio_data")
                            timing.mark("tts_first_audio_chunk")
                        yield chunk
            except Exception:
                failed = True
                logger.exception("Streaming fallback TTS failed trace=%s", timing.trace_id)
            finally:
                timing.mark("tts_completion")
                timing.log("error" if failed else "complete")
                if timing_message:
                    _save_message_timing(timing_message.id, timing.snapshot())

        response = StreamingHttpResponse(no_emotion_audio_iterator(), content_type="audio/mpeg")
        response["Cache-Control"] = "no-store"
        response["X-Accel-Buffering"] = "no"
        return response

    voice = "onyx" if re.search(r"\bmale voice\b", emotion.lower()) else "coral"

    def audio_iterator():
        failed = False
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=dialogue,
                instructions=emotion,
                response_format="mp3",
                stream_format="audio",
            ) as response:
                for chunk in response.iter_bytes():
                    if chunk:
                        if "tts_first_audio_data" not in timing.events:
                            timing.mark("tts_first_audio_data")
                            timing.mark("tts_first_audio_chunk")
                        yield chunk
        except Exception:
            failed = True
            logger.exception("Streaming TTS failed trace=%s", timing.trace_id)
            return
        finally:
            timing.mark("tts_completion")
            timing.log("error" if failed else "complete")
            if timing_message:
                _save_message_timing(timing_message.id, timing.snapshot())

    response = StreamingHttpResponse(audio_iterator(), content_type="audio/mpeg")
    response["Cache-Control"] = "no-store"
    response["X-Accel-Buffering"] = "no"
    return response


@login_required
@require_POST
def student_voice_timing(request):
    """Receive and persist the browser's final marks for the conversation log."""
    if not (_is_student(request.user) or _is_professor(request.user)):
        return redirect("vip:home")
    payload = request.POST.get("timing")
    timing = VoicePipelineTiming(payload)
    timing.mark("client_report_received")
    timing.log("client_complete")
    message = None
    message_id = request.POST.get("message_id")
    if message_id:
        filters = {
            "pk": message_id,
            "sender": ChatMessage.Sender.ASSISTANT,
            "session__student": request.user,
        }
        if request.POST.get("session_id"):
            filters["session_id"] = request.POST.get("session_id")
        message = ChatMessage.objects.filter(**filters).first()
    if message is None:
        session_id = request.POST.get("session_id")
        turn_id = request.POST.get("turn_id")
        if session_id and turn_id:
            message = ChatMessage.objects.filter(
                session_id=session_id,
                session__student=request.user,
                sender=ChatMessage.Sender.ASSISTANT,
                turn_id=turn_id,
            ).first()
            if message is None:
                message = ChatMessage.objects.filter(
                    session_id=session_id,
                    session__student=request.user,
                    sender=ChatMessage.Sender.STUDENT,
                    turn_id=turn_id,
                ).first()
    if message is None:
        # A trace can arrive without identifiers from an older browser page.
        # Search only the current user's messages so this fallback cannot
        # update another user's conversation.
        for candidate in ChatMessage.objects.filter(session__student=request.user).order_by("-created_at"):
            if (candidate.pipeline_timing or {}).get("trace_id") == timing.trace_id:
                message = candidate
                break
    persisted = _save_message_timing(message.id, timing.snapshot()) if message else False
    return JsonResponse({"ok": True, "trace_id": timing.trace_id, "persisted": persisted})
