import os
import json
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vipson_manager.settings")

import django

django.setup()

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse

from vip.models import ChatMessage, ChatSession, RolePrompt
from vip.forms import RolePromptForm
from vip.views import (
    DEFAULT_INTRODUCTION,
    _conversation_messages,
    _accept_learner_turn,
    _merge_pipeline_timing,
    _release_learner_turn,
    _persist_assistant_response,
    _rendered_chat_messages,
    split_dialogue_and_voice,
    student_download_session,
    student_message_tts,
    student_voice_timing,
)
from vip.voice_timing import VoicePipelineTiming


class RelatedMessages:
    def __init__(self, messages):
        self.messages = messages

    def order_by(self, *_args):
        return self.messages


class ConversationViewTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.professor = get_user_model().objects.create_superuser(
            username="professor",
            email="professor@example.com",
            password="test-password",
        )
        cls.prompt = RolePrompt.objects.create(
            title="Rachel",
            content=(
                "## Role\nRachel, the daughter\n\n"
                "## Learner Role\nHome-health registered nurse\n\n"
                "## Introduction\nWelcome.\n\n"
                "## Opening Line\nI am worried.\n\n"
                "## Beginning\nI am overwhelmed.\n\n"
                "## Middle\nI ask questions.\n\n"
                "## Ending\nI am ready.\n\n"
                "## Closing\nThank you.\n\n"
                "## Voice Gender\nfemale\n\n"
                "## Voice Style\nworried\n"
            ),
            created_by=cls.professor,
            is_active=True,
        )

    def test_display_and_history_remove_only_assistant_voice_annotations(self):
        assistant = SimpleNamespace(
            id=1,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I am scared. [tense, anxious]",
            voice_metadata="female voice, tense",
            created_at=None,
        )
        learner = SimpleNamespace(
            id=2,
            sender=ChatMessage.Sender.STUDENT,
            content="I said [this] to Rachel.",
            voice_metadata="",
            created_at=None,
        )
        session = SimpleNamespace(messages=RelatedMessages([assistant, learner]))

        rendered = _rendered_chat_messages(session, "Student")
        self.assertEqual(len(rendered), 2)
        self.assertEqual(rendered[0]["display_content"], "I am scared.")
        self.assertTrue(rendered[0]["voice_enabled"])
        self.assertEqual(rendered[1]["display_content"], "I said [this] to Rachel.")
        self.assertFalse(rendered[1]["voice_enabled"])

        history = _conversation_messages([assistant, learner])
        self.assertEqual(
            history[0],
            {"role": "assistant", "content": "SIMULATED CHARACTER:\nI am scared."},
        )
        self.assertEqual(
            history[1],
            {"role": "user", "content": "LEARNER:\nI said [this] to Rachel."},
        )
        self.assertEqual(
            split_dialogue_and_voice("I am scared. [tense, anxious]"),
            ("I am scared.", "tense, anxious"),
        )

    @patch("vip.views._generate_assistant_response")
    def test_first_learner_message_and_completion_response_are_each_persisted_once(self, generate):
        generate.return_value = (
            "Thank you.",
            True,
            {
                "current_stage": "ending",
                "phase": "closure_preference",
                "voice_metadata": "female voice, calm",
            },
        )
        self.client.force_login(self.professor)
        response = self.client.post(
            reverse("vip:professor_test_chat"),
            {"action": "send_message", "prompt_id": self.prompt.id, "message": "Hello Rachel."},
        )
        self.assertEqual(response.status_code, 302)

        session = ChatSession.objects.get(student=self.professor, role_prompt=self.prompt)
        messages = list(session.messages.order_by("created_at"))
        self.assertEqual(
            [(message.sender, message.content) for message in messages],
            [
                (ChatMessage.Sender.ASSISTANT, "Welcome."),
                (ChatMessage.Sender.STUDENT, "Hello Rachel."),
                (ChatMessage.Sender.ASSISTANT, "Thank you."),
            ],
        )
        self.assertEqual(session.messages.filter(sender=ChatMessage.Sender.ASSISTANT).count(), 2)
        self.assertTrue(session.ended_at)

        rendered_response = self.client.get(response.url)
        self.assertEqual(rendered_response.status_code, 200)
        self.assertEqual(rendered_response.content.decode().count("Thank you."), 1)

    @patch("vip.views._generate_assistant_response")
    def test_retried_turn_id_does_not_generate_or_persist_a_second_response(self, generate):
        generate.return_value = (
            "I am listening.",
            False,
            {"current_stage": "middle", "phase": "normal", "voice_metadata": "female voice, calm"},
        )
        self.client.force_login(self.professor)
        payload = {
            "action": "send_message",
            "prompt_id": self.prompt.id,
            "message": "Please tell me more.",
            "turn_id": "retryable-turn-1",
        }

        first = self.client.post(reverse("vip:professor_test_chat"), payload)
        second = self.client.post(reverse("vip:professor_test_chat"), payload)

        self.assertEqual(first.status_code, 302)
        self.assertEqual(second.status_code, 302)
        session = ChatSession.objects.get(student=self.professor, role_prompt=self.prompt)
        self.assertEqual(session.messages.filter(sender=ChatMessage.Sender.STUDENT).count(), 1)
        self.assertEqual(session.messages.filter(sender=ChatMessage.Sender.ASSISTANT).count(), 2)
        self.assertEqual(generate.call_count, 1)

    def test_session_accepts_one_pending_turn_and_stale_commit_is_rejected(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        first_session, first_status, _ = _accept_learner_turn(session, "First.", "turn-1")
        self.assertEqual(first_status, "accepted")

        _, pending_status, _ = _accept_learner_turn(first_session, "Second.", "turn-2")
        self.assertEqual(pending_status, "pending")

        debug = {"current_stage": "middle", "phase": "normal", "voice_metadata": "female voice, calm"}
        self.assertTrue(_persist_assistant_response(first_session, "First response.", False, debug, "turn-1"))
        self.assertFalse(_persist_assistant_response(first_session, "Stale response.", False, debug, "turn-1"))
        self.assertEqual(
            ChatMessage.objects.filter(session=first_session, sender=ChatMessage.Sender.ASSISTANT).count(),
            1,
        )

    def test_objective_progress_is_persisted_with_the_session(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "I am ready to continue.", "objective-turn")
        self.assertEqual(status, "accepted")

        debug = {
            "current_stage": "middle",
            "phase": "normal",
            "voice_metadata": "female voice, calm",
            "active_objective": "next-steps",
            "covered_objectives": ["situation"],
            "unresolved_objectives": ["next-steps"],
            "recent_topics": ["I am ready to continue."],
            "ending_ready": True,
        }
        self.assertTrue(_persist_assistant_response(session, "I hear you.", False, debug, "objective-turn"))

        session.refresh_from_db()
        self.assertEqual(session.active_objective, "next-steps")
        self.assertEqual(session.covered_objectives, ["situation"])
        self.assertEqual(session.unresolved_objectives, ["next-steps"])
        self.assertEqual(session.recent_topics, ["I am ready to continue."])
        self.assertTrue(session.ending_ready)

    def test_prompt_form_round_trips_generic_objective_configuration(self):
        content = self.prompt.content + """

## Conversation Objectives
### Understand the situation
Possible expressions:
- What happened?
Resolved when:
The learner has explained the situation.
"""
        initial = RolePromptForm.initial_from_content(content, title="Objective prompt", is_active=True)
        form = RolePromptForm(initial)

        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["introduction"], "Welcome.")
        rendered = form.render_markdown_content()
        self.assertIn("## Introduction", rendered)
        self.assertIn("Welcome.", rendered)
        self.assertIn("## Conversation Goals", rendered)
        self.assertIn("### Understand the situation", rendered)
        self.assertIn("Resolved when:", rendered)

    def test_failed_turn_can_be_retried_without_duplicate_learner_message(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "Try again.", "retry-turn")
        self.assertEqual(status, "accepted")
        self.assertTrue(_release_learner_turn(session, "retry-turn"))

        session, status, _ = _accept_learner_turn(session, "Try again.", "retry-turn")
        self.assertEqual(status, "accepted")
        self.assertEqual(
            ChatMessage.objects.filter(session=session, sender=ChatMessage.Sender.STUDENT).count(),
            1,
        )

        debug = {"current_stage": "middle", "phase": "normal", "voice_metadata": "female voice, calm"}
        self.assertTrue(_persist_assistant_response(session, "Recovered.", False, debug, "retry-turn"))
        session.refresh_from_db()
        self.assertEqual(session.active_turn_id, "")
        self.assertEqual(session.last_completed_turn_id, "retry-turn")

    def test_late_cleanup_cannot_release_a_replacement_claim(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "Retry claim.", "claim-turn")
        self.assertEqual(status, "accepted")
        first_claim = session.active_claim_id
        self.assertTrue(_release_learner_turn(session, "claim-turn", first_claim))

        session, status, _ = _accept_learner_turn(session, "Retry claim.", "claim-turn")
        self.assertEqual(status, "accepted")
        second_claim = session.active_claim_id
        self.assertNotEqual(first_claim, second_claim)
        self.assertFalse(_release_learner_turn(session, "claim-turn", first_claim))
        session.refresh_from_db()
        self.assertEqual(session.active_claim_id, second_claim)

    def test_stale_claim_cannot_commit_after_a_retry_reclaims_the_turn(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "Retry response.", "stale-turn")
        self.assertEqual(status, "accepted")
        first_claim = session.active_claim_id
        self.assertTrue(_release_learner_turn(session, "stale-turn", first_claim))
        session, status, _ = _accept_learner_turn(session, "Retry response.", "stale-turn")
        self.assertEqual(status, "accepted")
        second_claim = session.active_claim_id

        debug = {"current_stage": "middle", "phase": "normal", "voice_metadata": "female voice, calm"}
        self.assertFalse(
            _persist_assistant_response(
                session,
                "Stale response.",
                False,
                debug,
                turn_id="stale-turn",
                claim_id=first_claim,
            )
        )
        self.assertEqual(
            ChatMessage.objects.filter(session=session, sender=ChatMessage.Sender.ASSISTANT).count(),
            0,
        )
        self.assertTrue(
            _persist_assistant_response(
                session,
                "Current response.",
                False,
                debug,
                turn_id="stale-turn",
                claim_id=second_claim,
            )
        )

    def test_cancelled_response_cannot_be_persisted(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "Cancel this response.", "cancelled-turn")
        self.assertEqual(status, "accepted")
        claim_id = session.active_claim_id

        self.assertFalse(
            _persist_assistant_response(
                session,
                "This must not be saved.",
                False,
                {"current_stage": "middle", "phase": "normal", "voice_metadata": ""},
                turn_id="cancelled-turn",
                cancellation_callback=lambda: True,
                claim_id=claim_id,
            )
        )
        self.assertFalse(
            ChatMessage.objects.filter(
                session=session,
                sender=ChatMessage.Sender.ASSISTANT,
                turn_id="cancelled-turn",
            ).exists()
        )
        self.assertTrue(_release_learner_turn(session, "cancelled-turn", claim_id))

    def test_reused_turn_id_with_different_content_is_rejected(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        _accept_learner_turn(session, "Original content.", "fixed-turn")
        self.assertEqual(
            _accept_learner_turn(session, "Changed content.", "fixed-turn")[1],
            "conflict",
        )

    def test_empty_completed_turn_is_idempotent(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        session, status, _ = _accept_learner_turn(session, "Stop now.", "stop-turn")
        self.assertEqual(status, "accepted")
        debug = {"current_stage": "ending", "phase": "normal", "voice_metadata": ""}
        self.assertTrue(_persist_assistant_response(session, "", True, debug, "stop-turn"))
        session.refresh_from_db()

        session, status, assistant = _accept_learner_turn(session, "Stop now.", "stop-turn")
        self.assertEqual(status, "duplicate")
        self.assertIsNone(assistant)

    @patch("vip.views._generate_assistant_response", side_effect=RuntimeError("generation failed"))
    def test_generation_error_releases_pending_turn(self, _generate):
        self.client.force_login(self.professor)
        with self.assertRaises(RuntimeError):
            self.client.post(
                reverse("vip:professor_test_chat"),
                {
                    "action": "send_message",
                    "prompt_id": self.prompt.id,
                    "message": "This should be retryable.",
                    "turn_id": "error-turn",
                },
            )

        session = ChatSession.objects.get(student=self.professor, role_prompt=self.prompt)
        self.assertEqual(session.active_turn_id, "")

    @patch("vip.views._generate_assistant_response")
    def test_new_chat_persists_and_displays_text_only_introduction_before_first_learner_message(self, generate):
        self.client.force_login(self.professor)
        response = self.client.post(
            reverse("vip:professor_test_chat"),
            {"action": "new_session", "prompt_id": self.prompt.id},
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("session=", response.url)
        self.assertFalse(generate.called)
        session = ChatSession.objects.get(student=self.professor, role_prompt=self.prompt)
        messages = list(session.messages.order_by("created_at"))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender, ChatMessage.Sender.ASSISTANT)
        self.assertEqual(messages[0].content, "Welcome.")
        self.assertEqual(messages[0].voice_metadata, "")

        page = self.client.get(response.url)
        self.assertEqual(page.status_code, 200)
        html = page.content.decode()
        self.assertEqual(html.count("Welcome."), 1)
        self.assertEqual(html.count('class="tts-button"'), 0)

    def test_new_chat_seeds_fallback_when_prompt_has_no_introduction(self):
        prompt = RolePrompt.objects.create(
            title="No Introduction",
            content=self.prompt.content.replace("## Introduction\nWelcome.\n", "## Introduction\n\n"),
            created_by=self.professor,
            is_active=True,
        )
        self.client.force_login(self.professor)

        response = self.client.post(
            reverse("vip:professor_test_chat"),
            {"action": "new_session", "prompt_id": prompt.id},
        )

        session = ChatSession.objects.get(student=self.professor, role_prompt=prompt)
        self.assertEqual(session.messages.first().content, DEFAULT_INTRODUCTION)
        self.assertNotEqual(session.messages.first().content, "I am worried.")
        self.assertEqual(response.status_code, 302)

    def test_chat_form_uses_explicit_action_url_without_streaming_interception(self):
        self.client.force_login(self.professor)
        response = self.client.get(reverse("vip:professor_test_chat"))
        html = response.content.decode()
        self.assertIn('class="message-form" method="post" action="/professor/test-chat/"', html)
        self.assertNotIn("ReadableStream", html)
        self.assertNotIn("streamVoiceConversation", html)
        self.assertNotIn("streaming-voice-status", html)

    def test_prompt_form_exposes_introduction_tab_and_field(self):
        self.client.force_login(self.professor)
        response = self.client.get(reverse("vip:create_prompt"))
        html = response.content.decode()
        self.assertIn('data-tab="introduction-tab"', html)
        self.assertIn('id="introduction-tab"', html)
        self.assertIn('id="id_introduction"', html)

    def test_prompt_form_saves_introduction_as_a_role_prompt_section(self):
        self.client.force_login(self.professor)
        response = self.client.post(
            reverse("vip:create_prompt"),
            {
                "title": "Introduction prompt",
                "role": "A simulated character",
                "background_context": "Relevant scenario context.",
                "learner_role": "The learner",
                "introduction": "This message appears when the chat starts.",
                "conversation_objectives": "### Goal 1\nUnderstand the situation.",
                "voice_gender": "female",
                "voice_style": "calm",
                "opening_line": "Hello.",
                "beginning": "Begin naturally.",
                "middle": "Continue naturally.",
                "ending": "Wrap up naturally.",
                "closing": "Goodbye.",
            },
        )

        self.assertEqual(response.status_code, 302)
        prompt = RolePrompt.objects.get(title="Introduction prompt")
        self.assertIn("## Introduction\nThis message appears when the chat starts.", prompt.content)

    @patch("openai.OpenAI")
    @patch("vip.views._load_api_key_from_txt", return_value="test-key")
    def test_tts_receives_dialogue_and_metadata_as_separate_values(self, _api_key, openai_client):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        message = ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I am scared. [tense, anxious]",
            voice_metadata="female voice, tense",
        )
        openai_client.return_value.audio.speech.create.return_value.read.return_value = b"audio"
        request = RequestFactory().get("/student/messages/1/tts/")
        request.user = self.professor

        response = student_message_tts(request, message.id)

        self.assertEqual(response.status_code, 200)
        kwargs = openai_client.return_value.audio.speech.create.call_args.kwargs
        self.assertEqual(kwargs["input"], "I am scared.")
        self.assertEqual(kwargs["instructions"], "female voice, tense")
        self.assertEqual(response.content, b"audio")

    @patch("vip.views._load_api_key_from_txt", return_value="test-key")
    @patch("openai.OpenAI")
    def test_text_only_introduction_does_not_generate_audio(self, openai_client, _api_key):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        message = ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="Welcome.",
        )
        request = RequestFactory().get("/student/messages/1/tts/")
        request.user = self.professor

        response = student_message_tts(request, message.id)

        self.assertEqual(response.status_code, 400)
        openai_client.assert_not_called()

    def test_download_includes_stored_and_legacy_voice_metadata(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I am scared.",
            voice_metadata="female voice, tense, anxious",
        )
        ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I need a moment. [female voice, tired]",
        )
        request = RequestFactory().get(reverse("vip:student_download_session", args=[session.id]))
        request.user = self.professor

        response = student_download_session(request, session.id)

        self.assertEqual(response.status_code, 200)
        log = response.content.decode()
        self.assertIn("Assistant: I am scared.", log)
        self.assertIn("Voice metadata: female voice, tense, anxious", log)
        self.assertIn("Assistant: I need a moment.", log)
        self.assertIn("Voice metadata: female voice, tired", log)

    def test_download_includes_persisted_pipeline_timestamps(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I am listening.",
            voice_metadata="female voice, calm",
            pipeline_timing={
                "trace_id": "download-trace",
                "client_events": {"stt_start": 1000.0},
                "server_events": {"gpt_request_start": {"epoch_ms": 1100.0}},
                "durations_ms": {"stt_completion_to_gpt_request_start": 100.0},
                "llm_context": [{"request_kind": "generation", "serialized_bytes": 1234}],
            },
        )
        request = RequestFactory().get(reverse("vip:student_download_session", args=[session.id]))
        request.user = self.professor

        response = student_download_session(request, session.id)

        log = response.content.decode()
        self.assertIn("Voice pipeline timing:", log)
        self.assertIn("download-trace", log)
        self.assertIn("stt_start", log)
        self.assertIn("gpt_request_start", log)
        self.assertIn("stt_completion_to_gpt_request_start", log)
        self.assertIn("LLM context measurements:", log)
        self.assertIn('"serialized_bytes": 1234', log)

    def test_voice_timing_report_is_saved_to_the_assistant_message(self):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        message = ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="I am listening.",
            turn_id="timed-turn",
            pipeline_timing={
                "trace_id": "timed-turn-trace",
                "server_events": {"gpt_request_start": {"epoch_ms": 1000.0}},
            },
        )
        request = RequestFactory().post(
            reverse("vip:student_voice_timing"),
            {
                "message_id": message.id,
                "timing": json.dumps(
                    {
                        "trace_id": "timed-turn-trace",
                        "events": {"audio_playback_start": 1200.0},
                    }
                ),
            },
        )
        request.user = self.professor

        response = student_voice_timing(request)

        self.assertEqual(response.status_code, 200)
        message.refresh_from_db()
        self.assertIn("gpt_request_start", message.pipeline_timing["server_events"])
        self.assertEqual(message.pipeline_timing["client_events"]["audio_playback_start"], 1200.0)
        self.assertIn("client_report_received", message.pipeline_timing["server_events"])

    def test_typed_turn_generates_complete_response_and_redirects_for_audio(self):
        self.client.force_login(self.professor)
        with patch("vip.views._generate_assistant_response") as generate:
            generate.return_value = (
                "A typed response.",
                False,
                {"current_stage": "middle", "phase": "normal", "voice_metadata": "female voice, calm"},
            )
            response = self.client.post(
                reverse("vip:professor_test_chat"),
                {"action": "send_message", "prompt_id": self.prompt.id, "message": "Typed input."},
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("autoplay=1", response.url)
        session = ChatSession.objects.get(student=self.professor, role_prompt=self.prompt)
        self.assertEqual(session.messages.order_by("created_at").last().content, "A typed response.")

    @patch("gtts.gTTS")
    @patch("vip.views._load_api_key_from_txt", return_value="test-key")
    def test_no_emotion_tts_returns_complete_audio(self, _api_key, gtts_client):
        session = ChatSession.objects.create(student=self.professor, role_prompt=self.prompt)
        message = ChatMessage.objects.create(
            session=session,
            sender=ChatMessage.Sender.ASSISTANT,
            content="A calm sentence.",
        )
        gtts_client.return_value.write_to_fp.side_effect = lambda fp: fp.write(b"plain-audio")
        request = RequestFactory().get(
            "/student/messages/1/tts/",
            {"emotion": "0", "trace": "plain-trace"},
        )
        request.user = self.professor

        response = student_message_tts(request, message.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"plain-audio")

    def test_pipeline_timing_merge_preserves_the_first_server_mark(self):
        merged = _merge_pipeline_timing(
            {
                "trace_id": "trace",
                "server_events": {"tts_request_start": {"epoch_ms": 1000.0}},
                "llm_context": [{"request_kind": "generation", "message_count": 3}],
            },
            {
                "trace_id": "trace",
                "server_events": {"tts_request_start": {"epoch_ms": 2000.0}},
                "client_events": {"audio_playback_start": 2100.0},
            },
        )
        self.assertEqual(merged["server_events"]["tts_request_start"]["epoch_ms"], 1000.0)
        self.assertEqual(merged["client_events"]["audio_playback_start"], 2100.0)
        self.assertEqual(merged["llm_context"][0]["message_count"], 3)

    def test_timing_snapshot_derives_cross_stage_latency(self):
        timing = VoicePipelineTiming(
            {
                "trace_id": "trace",
                "events": {
                    "stt_start": 900.0,
                    "recording_end": 1000.0,
                    "stt_completion": 1100.0,
                    "tts_text_first_usable": 1300.0,
                    "tts_request_start": 1400.0,
                    "first_tts_audio_data": 1500.0,
                    "audio_playback_start": 1550.0,
                    "client_pipeline_complete": 1700.0,
                },
                "server_events": {
                    "gpt_request_start": {"epoch_ms": 1200.0},
                    "gpt_first_output": {"epoch_ms": 1250.0},
                    "response_complete": {"epoch_ms": 1600.0},
                },
            }
        )
        durations = timing.snapshot()["durations_ms"]
        self.assertEqual(durations["stt_start_to_stt_completion"], 200.0)
        self.assertEqual(durations["recording_end_to_stt_completion"], 100.0)
        self.assertEqual(durations["stt_completion_to_gpt_request_start"], 100.0)
        self.assertEqual(durations["stt_completion_to_gpt_first_output"], 150.0)
        self.assertEqual(durations["gpt_first_output_to_first_usable_tts_text"], 50.0)
        self.assertEqual(durations["first_usable_tts_text_to_tts_request"], 100.0)
        self.assertEqual(durations["tts_request_to_first_audio_data"], 100.0)
        self.assertEqual(durations["first_audio_data_to_playback_start"], 50.0)
        self.assertEqual(durations["gpt_first_output_to_audio_playback"], 300.0)
        self.assertEqual(durations["recording_end_to_first_audible_response"], 550.0)
        self.assertEqual(durations["response_generation_to_client_pipeline_complete"], 100.0)
