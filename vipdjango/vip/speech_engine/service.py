"""Server-only helpers for ElevenLabs Speech Engine browser sessions."""

import os
import logging
import time
from dataclasses import dataclass

from django.db import IntegrityError


class SpeechEngineConfigurationError(RuntimeError):
    """Raised when voice was requested before its server configuration exists."""


@dataclass(frozen=True)
class VoiceToken:
    token: str
    provider_conversation_id: str = ""


@dataclass(frozen=True)
class ElevenLabsVoiceOption:
    """A selectable voice and its provider-hosted sample, when supplied."""

    voice_id: str
    name: str
    preview_url: str = ""


logger = logging.getLogger(__name__)
_VOICE_CACHE = {"expires": 0.0, "options": ()}

# These events support the browser call UI and diagnostics.  The browser uses
# the SDK's speaking-to-listening mode transition as the audio-drained boundary;
# ``agent_response_complete`` describes provider response processing and is not
# itself a playback-complete signal.
SPEECH_ENGINE_CLIENT_EVENTS = (
    "audio",
    "interruption",
    "user_transcript",
    "tentative_user_transcript",
    "agent_response",
    "agent_response_correction",
    "agent_response_complete",
    "agent_tool_response",
)
# The configured external engine currently uses ElevenLabs' 10-minute
# provider session cap. Keep the browser's expected-limit UI aligned unless a
# deployment deliberately sets a different explicit cap in its environment.
_DEFAULT_MAX_DURATION_SECONDS = 600


def speech_engine_id():
    return os.getenv("ELEVENLABS_SPEECH_ENGINE_ID", "").strip()


def api_key():
    return os.getenv("ELEVENLABS_API_KEY", "").strip()


def speech_engine_max_duration_seconds():
    """Return the provider session limit used for managed voice resources."""
    raw_value = os.getenv("ELEVENLABS_SPEECH_ENGINE_MAX_DURATION_SECONDS", "").strip()
    if not raw_value:
        return _DEFAULT_MAX_DURATION_SECONDS
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid ELEVENLABS_SPEECH_ENGINE_MAX_DURATION_SECONDS=%r; using %s.",
            raw_value,
            _DEFAULT_MAX_DURATION_SECONDS,
        )
        return _DEFAULT_MAX_DURATION_SECONDS
    if value < 60:
        logger.warning(
            "ELEVENLABS_SPEECH_ENGINE_MAX_DURATION_SECONDS must be at least 60; using %s.",
            _DEFAULT_MAX_DURATION_SECONDS,
        )
        return _DEFAULT_MAX_DURATION_SECONDS
    return value


def speech_engine_turn_config():
    """Keep silence from ending a live training conversation."""
    return {"silence_end_call_timeout": -1}


def speech_engine_conversation_config():
    """Request the client events and session duration required by this UI."""
    return {
        "max_duration_seconds": speech_engine_max_duration_seconds(),
        "client_events": list(SPEECH_ENGINE_CLIENT_EVENTS),
    }


def is_speech_engine_configured():
    """Whether Django can safely issue a browser-only conversation token."""
    return bool(speech_engine_id() and api_key())


def list_elevenlabs_voice_options():
    """Return live ElevenLabs voices, including provider-hosted samples."""
    if _VOICE_CACHE["expires"] > time.monotonic():
        return list(_VOICE_CACHE["options"])
    elevenlabs_api_key = api_key()
    if not elevenlabs_api_key:
        return []
    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=elevenlabs_api_key)
        options = []
        next_page_token = None
        while True:
            page = client.voices.search(
                next_page_token=next_page_token,
                page_size=100,
                sort="name",
                sort_direction="asc",
            )
            for voice in getattr(page, "voices", ()):
                voice_id = (getattr(voice, "voice_id", "") or "").strip()
                name = (getattr(voice, "name", "") or "").strip()
                if voice_id and name:
                    options.append(
                        ElevenLabsVoiceOption(
                            voice_id=voice_id,
                            name=name,
                            preview_url=(getattr(voice, "preview_url", "") or "").strip(),
                        )
                    )
            if not getattr(page, "has_more", False):
                break
            next_page_token = getattr(page, "next_page_token", None)
            if not next_page_token:
                break
        options = sorted(
            {option.voice_id: option for option in options}.values(),
            key=lambda option: option.name.casefold(),
        )
        _VOICE_CACHE.update(expires=time.monotonic() + 300, options=tuple(options))
        return options
    except Exception:
        logger.exception("Could not retrieve the ElevenLabs voice list")
        return []


def _speech_engine_for_voice(client, voice_id):
    """Get/create a fixed-voice Speech Engine resource for one selected voice.

    Speech Engine's current initiation overrides do not expose a per-call TTS
    voice override, so a resource is provisioned once per selected voice.
    All resources point at the same authenticated upstream adapter.
    """
    from ..models import SpeechEngineVoiceResource

    existing = SpeechEngineVoiceResource.objects.filter(voice_id=voice_id).first()
    if existing:
        return existing.speech_engine_id

    # ElevenLabs fixes a Speech Engine resource's TTS voice. The configured
    # external ID supplies the authenticated upstream settings; one provider
    # resource per selected roleplay voice preserves the prompt's voice choice
    # without mutating a shared engine during another learner's call.
    base = client.speech_engine.get(speech_engine_id())
    if not base.config or not base.config.speech_engine:
        raise RuntimeError("The configured Speech Engine did not return its upstream configuration.")
    created = client.speech_engine.create(
        name=f"Virtual Caretaker — {voice_id[:32]}",
        speech_engine=base.config.speech_engine,
        tts={
            "voice_id": voice_id,
            "model_id": "eleven_v3_conversational",
            "expressive_mode": True,
        },
        turn=speech_engine_turn_config(),
        conversation=speech_engine_conversation_config(),
        # The application never uses an ElevenLabs first message: narration
        # is completed locally before this live session is opened.
        overrides={"first_message": False},
        tags=["virtual-caretaker", "managed-voice"],
    )
    created_id = created.engine_id
    try:
        mapping = SpeechEngineVoiceResource.objects.create(
            voice_id=voice_id,
            speech_engine_id=created_id,
        )
    except IntegrityError:
        # Another web worker may create the same mapping at the same moment.
        # Use its durable resource instead of changing either active engine.
        mapping = SpeechEngineVoiceResource.objects.filter(voice_id=voice_id).first()
        if not mapping:
            raise
        logger.warning("A duplicate Speech Engine resource was created during a voice mapping race: %s", created_id)
    return mapping.speech_engine_id


def issue_webrtc_token(participant_name="", voice_id=""):
    """Issue a short-lived token using the official server-side SDK.

    The ElevenLabs API key and Speech Engine ID deliberately never leave this
    process.  The returned token is consumed directly by the browser SDK.
    """
    engine_id = speech_engine_id()
    elevenlabs_api_key = api_key()
    if not engine_id or not elevenlabs_api_key:
        raise SpeechEngineConfigurationError(
            "ELEVENLABS_API_KEY and ELEVENLABS_SPEECH_ENGINE_ID must be configured."
        )

    try:
        from elevenlabs import ElevenLabs
    except ImportError as error:  # pragma: no cover - deployment configuration
        raise SpeechEngineConfigurationError(
            "The elevenlabs package is not installed. Install vipdjango/requirements.txt."
        ) from error

    selected_voice_id = (voice_id or "").strip()
    if not selected_voice_id:
        raise SpeechEngineConfigurationError("A roleplay ElevenLabs voice must be selected.")

    client = ElevenLabs(api_key=elevenlabs_api_key)
    selected_engine_id = _speech_engine_for_voice(client, selected_voice_id)
    response = client.conversational_ai.conversations.get_webrtc_token(
        agent_id=selected_engine_id,
        participant_name=participant_name or None,
    )
    token = (getattr(response, "token", "") or "").strip()
    if not token:
        raise RuntimeError("ElevenLabs returned a token response without a token.")
    return VoiceToken(
        token=token,
        provider_conversation_id=(getattr(response, "conversation_id", "") or "").strip(),
    )


class ElevenLabsSpeechStream:
    """Small iterable wrapper used by Django's streaming narrator response."""

    def __init__(self, api_key_value, *, voice_id, text):
        try:
            from elevenlabs import ElevenLabs
        except ImportError as error:  # pragma: no cover - deployment configuration
            raise SpeechEngineConfigurationError("The elevenlabs package is not installed.") from error
        self.client = ElevenLabs(api_key=api_key_value)
        self.stream = self.client.text_to_speech.stream(
            voice_id=voice_id,
            text=text,
            model_id=os.getenv("ELEVENLABS_NARRATOR_MODEL", "eleven_v3"),
            output_format="mp3_44100_128",
        )

    def __iter__(self):
        return iter(self.stream)
