"""ElevenLabs Speech Engine adapter for the existing Django text chat.

This module intentionally contains no scenario prompt, LLM call, or parallel
conversation history.  It turns a finalized ElevenLabs transcript into the
same claimed learner turn used by the browser text form, then gives the saved
response back to ElevenLabs for speech synthesis.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
import hashlib
import logging
import os
import threading

from asgiref.sync import sync_to_async
from django.db import close_old_connections
from django.utils import timezone

from ..models import ChatSession, VoiceConversation
from ..views import (
    _accept_learner_turn,
    _conversation_messages,
    _finalize_voice_call,
    _generate_assistant_response,
    _persist_assistant_response,
    _release_learner_turn,
)
from .service import SpeechEngineConfigurationError, api_key, speech_engine_id

logger = logging.getLogger(__name__)


@dataclass
class ClaimedVoiceTurn:
    chat_session_id: int
    status: str
    turn_id: str
    claim_id: str = ""
    existing_assistant_text: str = ""


@dataclass
class ActiveVoiceTurn:
    cancel_event: threading.Event
    chat_session_id: int
    turn_id: str
    claim_id: str
    transcript_key: tuple = ()
    response_text: str = ""


@dataclass
class VoiceConnectionState:
    active_turn: ActiveVoiceTurn | None = None
    # The provider can replay an event after a newer event has arrived, so
    # remembering only the immediately preceding ID is insufficient.
    recent_provider_event_ids: set = field(default_factory=set)
    provider_event_order: deque = field(default_factory=lambda: deque(maxlen=64))
    completed_transcript_responses: dict = field(default_factory=dict)
    completed_transcript_order: deque = field(default_factory=lambda: deque(maxlen=32))
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _with_database_connection(function, *args):
    """Use fresh Django connections from the standalone async server threads."""
    close_old_connections()
    try:
        return function(*args)
    finally:
        close_old_connections()


async def _run_sync(function, *args):
    return await sync_to_async(_with_database_connection, thread_sensitive=False)(function, *args)


async def _release_voice_turn_safely(chat_session_id, turn_id, claim_id):
    """Finish a worker-thread release even if the provider cancels its task."""
    release_task = asyncio.create_task(
        _run_sync(_release_voice_turn, chat_session_id, turn_id, claim_id)
    )
    try:
        return await asyncio.shield(release_task)
    except asyncio.CancelledError:
        try:
            return await release_task
        finally:
            raise


def _voice_call_for_provider_id(provider_conversation_id):
    return (
        VoiceConversation.objects.select_related("chat_session", "chat_session__role_prompt")
        .filter(provider_conversation_id=provider_conversation_id)
        .first()
    )


def _mark_voice_connected(provider_conversation_id):
    call = _voice_call_for_provider_id(provider_conversation_id)
    if not call:
        return None
    if call.status != VoiceConversation.Status.ENDED:
        call.status = VoiceConversation.Status.CONNECTED
        call.connected_at = call.connected_at or timezone.now()
        call.failure_reason = ""
        call.save(update_fields=["status", "connected_at", "failure_reason"])
    return call


def _mark_voice_ended(provider_conversation_id, failed=False):
    call = _voice_call_for_provider_id(provider_conversation_id)
    if not call:
        return
    _finalize_voice_call(call, failed=failed)


def _socket_close_details(speech_session):
    """Read provider transport details when the SDK reports a dropped socket."""
    websocket = getattr(speech_session, "_ws", None)
    return (
        getattr(websocket, "close_code", None),
        (getattr(websocket, "close_reason", "") or "")[:240],
    )


def _claim_voice_turn(chat_session_id, text, turn_id):
    chat_session = ChatSession.objects.select_related("role_prompt").get(pk=chat_session_id)
    claimed_session, status, existing_assistant = _accept_learner_turn(
        chat_session,
        text,
        turn_id,
        allow_voice_supersede=True,
    )
    return ClaimedVoiceTurn(
        chat_session_id=claimed_session.id,
        status=status,
        turn_id=turn_id,
        claim_id=claimed_session.active_claim_id if status == "accepted" else "",
        existing_assistant_text=existing_assistant.content if existing_assistant else "",
    )


def _release_voice_turn(chat_session_id, turn_id, claim_id):
    chat_session = ChatSession.objects.get(pk=chat_session_id)
    return _release_learner_turn(chat_session, turn_id, claim_id)


def _generate_and_persist_voice_response(claim, cancellation_event):
    """Run the existing text path once and commit only a non-cancelled reply."""
    chat_session = ChatSession.objects.select_related("role_prompt").get(pk=claim.chat_session_id)
    if cancellation_event.is_set():
        logger.info("Voice turn cancelled before generation chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
        _release_learner_turn(chat_session, claim.turn_id, claim.claim_id)
        return ""

    logger.info("Voice turn generation started chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
    try:
        assistant_text, conversation_complete, debug_info = _generate_assistant_response(
            chat_session.scenario_content or getattr(chat_session.role_prompt, "content", ""),
            _conversation_messages(chat_session),
            chat_session,
        )
    except Exception:
        logger.exception("Voice ConversationEngine generation failed chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
        raise
    if cancellation_event.is_set():
        logger.info("Voice turn cancelled after generation chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
        _release_learner_turn(chat_session, claim.turn_id, claim.claim_id)
        return ""

    try:
        persisted = _persist_assistant_response(
            chat_session,
            assistant_text,
            conversation_complete,
            debug_info,
            turn_id=claim.turn_id,
            claim_id=claim.claim_id,
            cancellation_callback=cancellation_event.is_set,
        )
    except Exception:
        logger.exception("Voice turn persistence failed chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
        raise
    if not persisted:
        logger.warning("Voice turn response was not persisted chat_session=%s turn=%s", claim.chat_session_id, claim.turn_id)
        return ""
    logger.info(
        "Voice turn persisted chat_session=%s turn=%s complete=%s response_chars=%s",
        claim.chat_session_id,
        claim.turn_id,
        conversation_complete,
        len(assistant_text or ""),
    )
    # The model owns inline Eleven v3 tags inside assistant_text. Persist and
    # send that raw text unchanged; never prepend a separate default tag.
    return assistant_text


class SpeechEngineAdapter:
    """Lifecycle callbacks supplied directly to the official Python SDK."""

    def __init__(self):
        self._connections = {}

    async def _mapped_call(self, provider_conversation_id, attempts=25):
        """Wait briefly for the browser fallback bind when mapping arrives late."""
        for attempt in range(attempts):
            call = await _run_sync(_voice_call_for_provider_id, provider_conversation_id)
            if call:
                return call
            if attempt < attempts - 1:
                await asyncio.sleep(0.2)
        return None

    async def _claim_voice_turn_safely(self, chat_session_id, text, turn_id):
        """Do not abandon a DB claim when the provider cancels its callback."""
        claim_task = asyncio.create_task(
            _run_sync(_claim_voice_turn, chat_session_id, text, turn_id)
        )
        try:
            return await asyncio.shield(claim_task)
        except asyncio.CancelledError:
            # sync_to_async work may continue after the awaiting coroutine is
            # canceled. Reap its result, then release an accepted claim that
            # was never able to receive an ActiveVoiceTurn wrapper.
            try:
                claim = await claim_task
            except BaseException:
                claim = None
            if claim and claim.status == "accepted":
                await _release_voice_turn_safely(
                    claim.chat_session_id,
                    claim.turn_id,
                    claim.claim_id,
                )
            raise

    @staticmethod
    def _turn_id(provider_conversation_id, transcript, provider_event_id=None):
        user_turns = sum(1 for message in transcript if getattr(message, "role", "") == "user")
        conversation_hash = hashlib.sha256(provider_conversation_id.encode("utf-8")).hexdigest()[:32]
        turn_key = f"event_{provider_event_id}" if provider_event_id is not None else str(user_turns)
        return f"eleven_{conversation_hash}_{turn_key}"

    @staticmethod
    def _provider_event_id(speech_session):
        """Read the SDK's current event ID; the callback omits it from its arguments."""
        return getattr(speech_session, "_current_event_id", None)

    @staticmethod
    def _transcript_key(transcript):
        """Identify one provider utterance independently of provider revisions."""
        user_count = sum(1 for message in transcript if getattr(message, "role", "") == "user")
        latest_user_text = SpeechEngineAdapter._latest_user_text(transcript)
        normalized_text = " ".join(latest_user_text.split()).casefold()
        return user_count, normalized_text

    @staticmethod
    def _remember_provider_event(state, provider_event_id):
        if provider_event_id is None:
            return
        if len(state.provider_event_order) == state.provider_event_order.maxlen:
            state.recent_provider_event_ids.discard(state.provider_event_order[0])
        state.provider_event_order.append(provider_event_id)
        state.recent_provider_event_ids.add(provider_event_id)

    @staticmethod
    def _remember_completed_response(state, transcript_key, response_text):
        if not response_text:
            return
        if transcript_key not in state.completed_transcript_responses:
            if len(state.completed_transcript_order) == state.completed_transcript_order.maxlen:
                expired_key = state.completed_transcript_order.popleft()
                state.completed_transcript_responses.pop(expired_key, None)
            state.completed_transcript_order.append(transcript_key)
        state.completed_transcript_responses[transcript_key] = response_text

    @staticmethod
    def _latest_user_text(transcript):
        for message in reversed(transcript):
            if getattr(message, "role", "") == "user":
                content = (getattr(message, "content", "") or "").strip()
                if content:
                    return content
        return ""

    async def _cancel_active_turn(self, state):
        active = state.active_turn
        if not active:
            return
        active.cancel_event.set()
        # The database response may already have been committed when the
        # provider cancels the callback during its interruption handoff. Keep
        # it available so the replacement event can replay it instead of
        # generating a second assistant turn.
        self._remember_completed_response(state, active.transcript_key, active.response_text)
        # The server SDK cancels the coroutine as well. Releasing this claim immediately lets the newer finalized utterance claim the session; the old worker sees cancel_event before it can write a reply.
        await _release_voice_turn_safely(
            active.chat_session_id,
            active.turn_id,
            active.claim_id,
        )
        if state.active_turn is active:
            state.active_turn = None

    async def on_init(self, provider_conversation_id, speech_session):
        self._connections.setdefault(provider_conversation_id, VoiceConnectionState())
        call = await self._mapped_call(provider_conversation_id)
        if call:
            await _run_sync(_mark_voice_connected, provider_conversation_id)
            logger.info(
                "Speech Engine connected provider_conversation=%s chat_session=%s",
                provider_conversation_id,
                call.chat_session_id,
            )
        else:
            # Do not reject an authenticated ElevenLabs connection here: the browser has a fallback bind call after startSession resolves.
            logger.warning("Speech Engine init is awaiting voice mapping provider_conversation=%s", provider_conversation_id)

    async def on_transcript(self, transcript, speech_session):
        provider_conversation_id = speech_session.conversation_id
        if not provider_conversation_id:
            logger.error("Speech Engine transcript arrived before init")
            return
        # Record arrival before awaiting the fallback mapping lookup. A transcript that was emitted during narration must remain ineligible even if the browser reports narrator completion while that lookup is in flight.
        transcript_received_at = timezone.now()

        call = await self._mapped_call(provider_conversation_id)
        if not call:
            logger.error("No Django voice mapping for provider_conversation=%s", provider_conversation_id)
            await speech_session.send_response("This voice session could not be matched. Please end it and start again.")
            return
        if (
            call.introduction_completed_at is None
            or call.introduction_completed_at > transcript_received_at
        ):
            logger.info(
                "Ignoring finalized transcript before introduction completion provider_conversation=%s",
                provider_conversation_id,
            )
            return
        if call.is_muted:
            logger.info(
                "Ignoring finalized transcript while microphone is muted provider_conversation=%s",
                provider_conversation_id,
            )
            return

        user_text = self._latest_user_text(transcript)
        if not user_text:
            logger.warning("Ignoring Speech Engine transcript without a final user message provider_conversation=%s", provider_conversation_id)
            return
        state = self._connections.setdefault(provider_conversation_id, VoiceConnectionState())
        provider_event_id = self._provider_event_id(speech_session)
        transcript_key = self._transcript_key(transcript)
        logger.info(
            "Speech Engine finalized transcript provider_conversation=%s chat_session=%s event_id=%s messages=%s text_chars=%s",
            provider_conversation_id,
            call.chat_session_id,
            provider_event_id,
            len(transcript),
            len(user_text),
        )
        claim = None
        active = None
        try:
            # Serialize the cancel/claim/active assignment handoff. Without
            # this, event 490 can inspect the database before event 489 has
            # installed its in-memory active turn, then both callbacks race.
            async with state.turn_lock:
                if provider_event_id is not None and provider_event_id in state.recent_provider_event_ids:
                    logger.info(
                        "Ignoring duplicate Speech Engine transcript provider_conversation=%s event_id=%s",
                        provider_conversation_id,
                        provider_event_id,
                    )
                    return
                self._remember_provider_event(state, provider_event_id)

                previous_active = state.active_turn
                if previous_active and previous_active.transcript_key == transcript_key:
                    logger.info(
                        "Ignoring repeated Speech Engine transcript provider_conversation=%s event_id=%s",
                        provider_conversation_id,
                        provider_event_id,
                    )
                    return
                completed_response = state.completed_transcript_responses.get(transcript_key)
                if completed_response:
                    logger.info(
                        "Replaying completed Speech Engine response provider_conversation=%s event_id=%s",
                        provider_conversation_id,
                        provider_event_id,
                    )
                    await speech_session.send_response(completed_response)
                    return

                await self._cancel_active_turn(state)
                active = None
                turn_id = self._turn_id(provider_conversation_id, transcript, provider_event_id)

                # A cancelled worker has already released its claim. A short
                # retry is still useful if the upstream callback arrives at
                # exactly the same time as that cancellation.
                for attempt in range(12):
                    try:
                        claim = await self._claim_voice_turn_safely(
                            call.chat_session_id,
                            user_text,
                            turn_id,
                        )
                    except Exception:
                        logger.exception(
                            "Voice turn claim failed chat_session=%s turn=%s attempt=%s",
                            call.chat_session_id,
                            turn_id,
                            attempt + 1,
                        )
                        raise
                    if claim.status != "pending":
                        break
                    logger.info(
                        "Voice turn claim pending; retrying chat_session=%s turn=%s attempt=%s",
                        call.chat_session_id,
                        turn_id,
                        attempt + 1,
                    )
                    await asyncio.sleep(0.2)

                if claim is None or claim.status == "closed":
                    logger.info("Ignoring voice turn for closed chat_session=%s", call.chat_session_id)
                    return
                if claim.status == "conflict":
                    logger.warning("Rejected conflicting voice retry chat_session=%s turn=%s", call.chat_session_id, turn_id)
                    return
                if claim.status == "duplicate":
                    if claim.existing_assistant_text:
                        await speech_session.send_response(claim.existing_assistant_text)
                    return
                if claim.status != "accepted":
                    logger.warning("Voice turn remained pending chat_session=%s turn=%s", call.chat_session_id, turn_id)
                    return

                logger.info(
                    "Voice turn claimed chat_session=%s turn=%s event_id=%s",
                    claim.chat_session_id,
                    claim.turn_id,
                    provider_event_id,
                )
                active = ActiveVoiceTurn(
                    cancel_event=threading.Event(),
                    chat_session_id=claim.chat_session_id,
                    turn_id=claim.turn_id,
                    claim_id=claim.claim_id,
                    transcript_key=transcript_key,
                )
                state.active_turn = active

            assistant_text = await _run_sync(_generate_and_persist_voice_response, claim, active.cancel_event)
            active.response_text = assistant_text
            self._remember_completed_response(state, transcript_key, assistant_text)
            if assistant_text and not active.cancel_event.is_set():
                logger.info(
                    "Sending Speech Engine response chat_session=%s turn=%s response_chars=%s",
                    active.chat_session_id,
                    active.turn_id,
                    len(assistant_text),
                )
                try:
                    await speech_session.send_response(assistant_text)
                except Exception:
                    logger.exception(
                        "Speech Engine response dispatch failed chat_session=%s turn=%s",
                        active.chat_session_id,
                        active.turn_id,
                    )
                    raise
                logger.info("Speech Engine response dispatched chat_session=%s turn=%s", active.chat_session_id, active.turn_id)
        except asyncio.CancelledError:
            if active:
                active.cancel_event.set()
                self._remember_completed_response(state, active.transcript_key, active.response_text)
                await _release_voice_turn_safely(
                    active.chat_session_id,
                    active.turn_id,
                    active.claim_id,
                )
            elif claim and claim.status == "accepted":
                await _release_voice_turn_safely(
                    claim.chat_session_id,
                    claim.turn_id,
                    claim.claim_id,
                )
            raise
        except Exception:
            if active:
                active.cancel_event.set()
                await _release_voice_turn_safely(
                    active.chat_session_id,
                    active.turn_id,
                    active.claim_id,
                )
            elif claim and claim.status == "accepted":
                await _release_voice_turn_safely(
                    claim.chat_session_id,
                    claim.turn_id,
                    claim.claim_id,
                )
            logger.exception(
                "Speech Engine response generation failed chat_session=%s turn=%s",
                getattr(claim, "chat_session_id", None),
                getattr(claim, "turn_id", None),
            )
            await speech_session.send_response("I could not generate a response just now. Please try again.")
        finally:
            if state.active_turn is active:
                state.active_turn = None

    async def on_close(self, speech_session):
        provider_conversation_id = speech_session.conversation_id
        if not provider_conversation_id:
            return
        call = await _run_sync(_voice_call_for_provider_id, provider_conversation_id)
        state = self._connections.pop(provider_conversation_id, None)
        if state:
            await self._cancel_active_turn(state)
        await _run_sync(_mark_voice_ended, provider_conversation_id)
        logger.info(
            "Speech Engine close provider_conversation=%s voice_call=%s chat_session=%s",
            provider_conversation_id,
            getattr(call, "pk", None),
            getattr(call, "chat_session_id", None),
        )

    async def on_disconnect(self, speech_session):
        provider_conversation_id = speech_session.conversation_id
        if not provider_conversation_id:
            return
        close_code, close_reason = _socket_close_details(speech_session)
        call = await _run_sync(_voice_call_for_provider_id, provider_conversation_id)
        state = self._connections.pop(provider_conversation_id, None)
        if state:
            await self._cancel_active_turn(state)
        failed = close_code not in {1000, 1001}
        await _run_sync(_mark_voice_ended, provider_conversation_id, failed)
        log = logger.warning if failed else logger.info
        log(
            "Speech Engine disconnect provider_conversation=%s voice_call=%s chat_session=%s close_code=%s close_reason=%r",
            provider_conversation_id,
            getattr(call, "pk", None),
            getattr(call, "chat_session_id", None),
            close_code,
            close_reason,
        )

    async def on_error(self, error, speech_session):
        provider_conversation_id = getattr(speech_session, "conversation_id", None)
        logger.error(
            "Speech Engine protocol error provider_conversation=%s error=%r",
            provider_conversation_id,
            error,
        )
        if provider_conversation_id:
            call = await _run_sync(_voice_call_for_provider_id, provider_conversation_id)
            state = self._connections.pop(provider_conversation_id, None)
            if state:
                await self._cancel_active_turn(state)
            await _run_sync(_mark_voice_ended, provider_conversation_id, True)
            logger.error(
                "Speech Engine error context provider_conversation=%s voice_call=%s chat_session=%s",
                provider_conversation_id,
                getattr(call, "pk", None),
                getattr(call, "chat_session_id", None),
            )


async def serve_speech_engine(port=None, debug=False):
    """Run the SDK-managed, authenticated Speech Engine upstream server."""
    elevenlabs_api_key = api_key()
    engine_id = speech_engine_id()
    if not elevenlabs_api_key or not engine_id:
        raise SpeechEngineConfigurationError(
            "ELEVENLABS_API_KEY and ELEVENLABS_SPEECH_ENGINE_ID must be configured."
        )
    try:
        from elevenlabs import AsyncElevenLabs
    except ImportError as error:  # pragma: no cover - deployment configuration
        raise SpeechEngineConfigurationError(
            "The elevenlabs package is not installed. Install vipdjango/requirements.txt."
        ) from error

    adapter = SpeechEngineAdapter()
    client = AsyncElevenLabs(api_key=elevenlabs_api_key)
    engine = await client.speech_engine.get(engine_id)
    logger.info("Starting Speech Engine adapter port=%s path=/ws engine=%s", port or int(os.getenv("SPEECH_ENGINE_PORT", "8081")), engine_id)
    await engine.serve(
        port=port or int(os.getenv("SPEECH_ENGINE_PORT", "8081")),
        path="/ws",
        debug=debug,
        on_init=adapter.on_init,
        on_transcript=adapter.on_transcript,
        on_close=adapter.on_close,
        on_disconnect=adapter.on_disconnect,
        on_error=adapter.on_error,
    )