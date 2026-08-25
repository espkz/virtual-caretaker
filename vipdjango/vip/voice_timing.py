"""Small, request-scoped timing helpers for the browser voice pipeline."""

import json
import logging
import time
import uuid


logger = logging.getLogger(__name__)

_FIRST_MARKS = {
    "recording_start",
    "recording_end",
    "speech_end",
    "stt_start",
    "stt_completion",
    "stt_finalization",
    "gpt_request_start",
    "gpt_first_output",
    "tts_text_first_usable",
    "response_display",
    "response_generation_complete",
    "response_complete",
    "client_pipeline_complete",
    "first_tts_audio_data",
    "tts_first_audio_chunk",
    "audio_playback_start",
    "tts_request_start",
}


class VoicePipelineTiming:
    """Collect client and server marks without making timing part of state."""

    def __init__(self, client_payload=None, trace_id=None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.started_at = time.perf_counter()
        self.events = {}
        self.client_events = {}
        self.add_client_payload(client_payload)
        self.mark("server_request_start")

    def add_client_payload(self, client_payload):
        if not client_payload:
            return
        try:
            payload = json.loads(client_payload) if isinstance(client_payload, str) else client_payload
        except (TypeError, ValueError):
            logger.warning("Invalid voice timing payload trace=%s", self.trace_id)
            return
        if not isinstance(payload, dict):
            return
        client_trace_id = payload.get("trace_id")
        if client_trace_id:
            self.trace_id = str(client_trace_id)[:96]
        marks = payload.get("events", payload)
        if isinstance(marks, dict):
            self.client_events.update(
                {
                    str(name): value
                    for name, value in marks.items()
                    if isinstance(value, (int, float))
                }
            )
        server_events = payload.get("server_events")
        if isinstance(server_events, dict):
            self.events.update(
                {
                    str(name): value
                    for name, value in server_events.items()
                    if isinstance(value, dict) and isinstance(value.get("epoch_ms"), (int, float))
                }
            )

    def mark(self, name):
        if name in _FIRST_MARKS and name in self.events:
            return
        now = time.perf_counter()
        self.events[name] = {
            "elapsed_ms": round((now - self.started_at) * 1000, 1),
            "epoch_ms": round(time.time() * 1000, 1),
        }

    def client_mark(self, name, value=None):
        """Record a browser mark received through the streaming channel."""
        self.mark(name)
        if isinstance(value, (int, float)):
            self.client_events[name] = value

    def snapshot(self):
        def epoch_ms(*names):
            for name in names:
                server_event = self.events.get(name)
                if isinstance(server_event, dict) and isinstance(server_event.get("epoch_ms"), (int, float)):
                    return server_event["epoch_ms"]
            for name in names:
                client_event = self.client_events.get(name)
                if isinstance(client_event, (int, float)):
                    return client_event
            return None

        def duration(start_names, end_names):
            if isinstance(start_names, str):
                start_names = (start_names,)
            if isinstance(end_names, str):
                end_names = (end_names,)
            start_ms = epoch_ms(*start_names)
            end_ms = epoch_ms(*end_names)
            if start_ms is None or end_ms is None:
                return None
            return round(end_ms - start_ms, 1)

        return {
            "trace_id": self.trace_id,
            "server_events": self.events,
            "client_events": self.client_events,
            "durations_ms": {
                "stt_start_to_stt_completion": duration("stt_start", ("stt_completion", "stt_finalization")),
                "recording_end_to_stt_completion": duration(
                    "recording_end", ("stt_completion", "stt_finalization")
                ),
                "stt_completion_to_gpt_request_start": duration("stt_completion", "gpt_request_start"),
                "stt_completion_to_gpt_first_output": duration("stt_completion", "gpt_first_output"),
                "gpt_first_output_to_first_usable_tts_text": duration(
                    "gpt_first_output", "tts_text_first_usable"
                ),
                "first_usable_tts_text_to_tts_request": duration(
                    "tts_text_first_usable", "tts_request_start"
                ),
                "gpt_first_output_to_tts_request": duration("gpt_first_output", "tts_request_start"),
                "tts_request_to_first_audio_data": duration(
                    "tts_request_start", ("first_tts_audio_data", "tts_first_audio_chunk", "tts_first_audio_data")
                ),
                "first_audio_data_to_playback_start": duration(
                    ("first_tts_audio_data", "tts_first_audio_chunk", "tts_first_audio_data"),
                    "audio_playback_start",
                ),
                "gpt_first_output_to_audio_playback": duration("gpt_first_output", "audio_playback_start"),
                "recording_end_to_first_visible_response": duration("recording_end", "response_display"),
                "recording_end_to_first_audible_response": duration("recording_end", "audio_playback_start"),
                "response_generation_to_client_pipeline_complete": duration(
                    ("response_generation_complete", "response_complete"), "client_pipeline_complete"
                ),
            },
        }

    def log(self, outcome="complete"):
        logger.info(
            "voice_pipeline trace=%s outcome=%s timings=%s",
            self.trace_id,
            outcome,
            json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True),
        )


def client_timing_payload(trace_id=None):
    """Return the initial browser-side payload shape used by the templates."""
    return {"trace_id": trace_id or uuid.uuid4().hex, "events": {}}
