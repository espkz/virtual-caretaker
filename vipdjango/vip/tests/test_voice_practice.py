import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from asgiref.sync import async_to_sync
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import SimpleTestCase, TestCase, TransactionTestCase
from django.urls import reverse
from django.utils import timezone

from vip import views
from vip.models import ChatMessage, ChatSession, RolePrompt, SpeechEngineVoiceResource, VoiceConversation
from vip.conversation_scenario import parse_scenario_prompt
from vip.forms import RolePromptForm
from vip.speech_engine import adapter as speech_engine_adapter
from vip.speech_engine.adapter import SpeechEngineAdapter, serve_speech_engine
from vip.speech_engine.service import (
    VoiceToken,
    issue_webrtc_token,
    list_elevenlabs_voice_options,
)


def scenario_text():
    content = (Path(settings.BASE_DIR).parent / "prompts" / "role_rachel_ellison_1.md").read_text(encoding="utf-8")
    return ("## Simulation Mode\nroleplay\n\n" + content).replace(
        "## Introduction Voice\n\n",
        "## Introduction Voice\nvoice_intro_test\n\n",
    ).replace(
        "## Roleplay Voice\n\n",
        "## Roleplay Voice\nvoice_roleplay_test\n\n",
    )


class VoicePracticeViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("voice-learner", password="test")
        self.user.groups.add(Group.objects.create(name="Class: Voice"))
        self.client.force_login(self.user)
        self.prompt = RolePrompt.objects.create(title="Voice scenario", content=scenario_text(), is_active=True)
        self.chat_session = views._create_chat_session(
            self.user,
            self.prompt,
            ChatSession.InteractionMode.VOICE,
        )

    def post_json(self, url_name, payload):
        return self.client.post(
            reverse(url_name),
            data=json.dumps(payload),
            content_type="application/json",
        )

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    @patch("vip.views.issue_webrtc_token", return_value=VoiceToken("browser-token", "conv_voice_123"))
    def test_token_creates_explicit_provider_mapping_and_returns_only_browser_data(self, issue_token):
        response = self.post_json(
            "vip:student_voice_token",
            {"prompt_id": self.prompt.id, "session_id": self.chat_session.id},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["token"], "browser-token")
        self.assertEqual(data["chat_session_id"], self.chat_session.id)
        self.assertEqual(data["max_duration_seconds"], 600)
        self.assertIn("introduction", data)
        self.assertEqual(data["character_label"], "Rachel Ellison")
        self.assertEqual(
            data["introduction_audio_url"],
            reverse("vip:student_voice_introduction", args=[self.chat_session.id]),
        )
        self.assertNotIn("ELEVENLABS_API_KEY", data)
        self.assertNotIn("seng_test", response.content.decode())
        voice_call = VoiceConversation.objects.get(pk=data["voice_call_id"])
        self.assertEqual(voice_call.chat_session_id, self.chat_session.id)
        self.assertEqual(voice_call.provider_conversation_id, "conv_voice_123")
        self.assertEqual(response["Cache-Control"], "no-store")
        issue_token.assert_called_once_with(
            participant_name=self.user.username,
            voice_id="voice_roleplay_test",
        )

    def test_late_provider_mapping_is_limited_to_the_owning_student_voice_call(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        response = self.post_json(
            "vip:student_voice_bind",
            {"voice_call_id": str(voice_call.pk), "provider_conversation_id": "conv_late_123"},
        )
        self.assertEqual(response.status_code, 200)
        voice_call.refresh_from_db()
        self.assertEqual(voice_call.provider_conversation_id, "conv_late_123")

        other = get_user_model().objects.create_user("other-voice-learner", password="test")
        other.groups.add(self.user.groups.first())
        self.client.force_login(other)
        response = self.post_json(
            "vip:student_voice_bind",
            {"voice_call_id": str(voice_call.pk), "provider_conversation_id": "conv_takeover"},
        )
        self.assertEqual(response.status_code, 404)

    def test_connection_diagnostic_records_sanitized_browser_and_session_context(self):
        voice_call = VoiceConversation.objects.create(
            chat_session=self.chat_session,
            provider_conversation_id="conv_server_123",
        )

        with self.assertLogs("vip.views", level="WARNING") as logs:
            response = self.post_json(
                "vip:student_voice_connection_diagnostic",
                {
                    "voice_call_id": str(voice_call.pk),
                    "event": "disconnect",
                    "connection_state": "listening",
                    "close_code": 1006,
                    "close_reason": "transport closed",
                    "provider_conversation_id": "conv_browser_456",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        entry = "\n".join(logs.output)
        self.assertIn("event=disconnect", entry)
        self.assertIn("state=listening", entry)
        self.assertIn("close_code=1006", entry)
        self.assertIn("conv_server_123", entry)
        self.assertIn("conv_browser_456", entry)
        self.assertIn(str(voice_call.pk), entry)
        self.assertIn(str(self.chat_session.pk), entry)

    def test_introduction_completion_opens_input_once_for_owning_voice_call(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)

        first = self.post_json(
            "vip:student_voice_ready",
            {"voice_call_id": str(voice_call.pk)},
        )
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["introduction_complete"])
        voice_call.refresh_from_db()
        completed_at = voice_call.introduction_completed_at
        self.assertIsNotNone(completed_at)

        second = self.post_json(
            "vip:student_voice_ready",
            {"voice_call_id": str(voice_call.pk)},
        )
        self.assertEqual(second.status_code, 200)
        voice_call.refresh_from_db()
        self.assertEqual(voice_call.introduction_completed_at, completed_at)

        other = get_user_model().objects.create_user("other-ready-learner", password="test")
        other.groups.add(self.user.groups.first())
        self.client.force_login(other)
        response = self.post_json(
            "vip:student_voice_ready",
            {"voice_call_id": str(voice_call.pk)},
        )
        self.assertEqual(response.status_code, 404)

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_voice_session_uses_call_ui_without_text_composer_or_secret(self):
        response = self.client.get(
            reverse("vip:student_dashboard"),
            {"prompt": self.prompt.id, "session": self.chat_session.id},
        )
        self.assertContains(response, "Voice Conversation")
        self.assertContains(response, "voice_chat.js")
        self.assertContains(response, "Start Call")
        self.assertContains(response, "Mute")
        self.assertContains(response, "End Call")
        self.assertNotContains(response, 'id="voice-pause"')
        self.assertNotContains(response, 'id="voice-resume"')
        self.assertNotContains(response, 'id="student-message-input"')
        self.assertNotContains(response, "server-secret")

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_new_chat_shows_briefing_and_mode_choice_without_creating_session(self):
        before = ChatSession.objects.count()
        response = self.client.get(
            reverse("vip:student_dashboard"),
            {"prompt": self.prompt.id, "new": 1},
        )
        self.assertContains(response, "How would you like to practice?")
        self.assertContains(response, "Conversation partner:")
        self.assertContains(response, "Start Voice Conversation")
        self.assertContains(response, "Start Text Conversation")
        self.assertEqual(ChatSession.objects.count(), before)

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_text_mode_is_fixed_and_never_loads_speech_engine_controls(self):
        response = self.client.post(
            reverse("vip:student_dashboard") + f"?prompt={self.prompt.id}&new=1",
            {"action": "start_session", "mode": "text", "prompt_id": self.prompt.id},
        )
        created = ChatSession.objects.latest("id")
        self.assertEqual(created.interaction_mode, ChatSession.InteractionMode.TEXT)
        self.assertRedirects(
            response,
            reverse("vip:student_dashboard") + f"?prompt={self.prompt.id}&session={created.id}",
            fetch_redirect_response=False,
        )
        page = self.client.get(response.url)
        self.assertContains(page, "Text Conversation")
        self.assertContains(page, "SIMULATION")
        self.assertContains(page, 'id="student-message-input"')
        self.assertNotContains(page, "voice_chat.js")
        self.assertNotContains(page, "ElevenLabsClient")
        self.assertNotContains(page, "Use expressive AI voice")
        self.assertNotContains(page, "Play responses aloud automatically")
        self.assertNotContains(page, 'id="voice-mute"')

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_non_javascript_voice_choice_creates_a_fixed_voice_session(self):
        response = self.client.post(
            reverse("vip:student_dashboard") + f"?prompt={self.prompt.id}&new=1",
            {"action": "start_session", "mode": "voice", "prompt_id": self.prompt.id},
        )
        created = ChatSession.objects.latest("id")
        self.assertEqual(created.interaction_mode, ChatSession.InteractionMode.VOICE)
        page = self.client.get(response.url)
        self.assertContains(page, "Voice Conversation")
        self.assertContains(page, "Start Call")
        self.assertNotContains(page, 'id="student-message-input"')

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_prompt_without_voice_ids_is_deactivated_and_cannot_be_tested(self):
        legacy_content = scenario_text().replace(
            "## Introduction Voice\nvoice_intro_test\n\n## Roleplay Voice\nvoice_roleplay_test\n\n",
            "## Voice Gender\nfemale\n\n",
        )
        legacy = RolePrompt.objects.create(title="Legacy scenario", content=legacy_content, is_active=True)
        self.assertFalse(legacy.is_active)
        picker = self.client.get(reverse("vip:student_dashboard"), {"prompt": legacy.id, "new": 1})
        self.assertIsNone(picker.context["selected_prompt"])
        self.assertNotContains(picker, "Legacy scenario")

        before = ChatSession.objects.count()
        text_response = self.client.post(
            reverse("vip:student_dashboard") + f"?prompt={legacy.id}&new=1",
            {"action": "start_session", "mode": "text", "prompt_id": legacy.id},
        )
        self.assertEqual(text_response.status_code, 200)
        self.assertEqual(ChatSession.objects.count(), before)

    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test"}, clear=False)
    def test_narrator_stream_uses_configured_introduction_voice(self):
        with patch("vip.views.ElevenLabsSpeechStream", return_value=iter([b"narration"])) as stream:
            response = self.client.get(
                reverse("vip:student_voice_introduction", args=[self.chat_session.id])
            )
            self.assertTrue(response.streaming)
            self.assertEqual(list(response.streaming_content), [b"narration"])
        kwargs = stream.call_args.kwargs
        self.assertEqual(kwargs["text"], parse_scenario_prompt(scenario_text()).introduction)
        self.assertEqual(kwargs["voice_id"], "voice_intro_test")

    def test_end_closes_the_mapped_chat_and_releases_an_inflight_claim(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        claimed, status, _ = views._accept_learner_turn(self.chat_session, "I am speaking", "voice-turn")
        self.assertEqual(status, "accepted")
        response = self.post_json("vip:student_voice_end", {"voice_call_id": str(voice_call.pk)})
        self.assertEqual(response.status_code, 200)
        voice_call.refresh_from_db()
        self.chat_session.refresh_from_db()
        self.assertEqual(voice_call.status, VoiceConversation.Status.ENDED)
        self.assertIsNotNone(voice_call.ended_at)
        self.assertIsNotNone(self.chat_session.ended_at)
        self.assertEqual(self.chat_session.active_turn_id, "")
        self.assertFalse(views._persist_assistant_response(claimed, "late reply", False, {}, "voice-turn", claim_id=claimed.active_claim_id))

    def test_voice_end_cannot_target_a_different_students_session(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        other = get_user_model().objects.create_user("other-end-learner", password="test")
        other.groups.add(self.user.groups.first())
        self.client.force_login(other)
        response = self.post_json("vip:student_voice_end", {"voice_call_id": str(voice_call.pk)})
        self.assertEqual(response.status_code, 404)
        voice_call.refresh_from_db()
        self.assertEqual(voice_call.status, VoiceConversation.Status.TOKEN_ISSUED)

    def test_voice_error_closes_the_saved_transcript_and_releases_an_inflight_claim(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        claimed, status, _ = views._accept_learner_turn(self.chat_session, "I am speaking", "failed-voice-turn")
        self.assertEqual(status, "accepted")

        response = self.post_json(
            "vip:student_voice_end",
            {"voice_call_id": str(voice_call.pk), "failed": True},
        )

        self.assertEqual(response.status_code, 200)
        voice_call.refresh_from_db()
        self.chat_session.refresh_from_db()
        self.assertEqual(voice_call.status, VoiceConversation.Status.ERROR)
        self.assertEqual(voice_call.failure_reason, "Voice connection ended unexpectedly.")
        self.assertIsNotNone(self.chat_session.ended_at)
        self.assertEqual(self.chat_session.active_turn_id, "")
        self.assertFalse(
            views._persist_assistant_response(
                claimed,
                "late reply",
                False,
                {},
                "failed-voice-turn",
                claim_id=claimed.active_claim_id,
            )
        )

    def test_mute_state_is_persisted_without_changing_conversation_state(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        muted = self.post_json(
            "vip:student_voice_mute",
            {"voice_call_id": str(voice_call.pk), "muted": True},
        )
        self.assertEqual(muted.status_code, 200)
        voice_call.refresh_from_db()
        self.assertTrue(voice_call.is_muted)
        self.assertIsNone(self.chat_session.ended_at)
        unmuted = self.post_json(
            "vip:student_voice_mute",
            {"voice_call_id": str(voice_call.pk), "muted": False},
        )
        self.assertEqual(unmuted.status_code, 200)
        voice_call.refresh_from_db()
        self.assertFalse(voice_call.is_muted)
        self.assertIsNone(self.chat_session.ended_at)

    def test_completion_endpoint_only_reports_existing_chat_state(self):
        voice_call = VoiceConversation.objects.create(chat_session=self.chat_session)
        open_response = self.post_json(
            "vip:student_voice_completion",
            {"voice_call_id": str(voice_call.pk)},
        )
        self.assertEqual(open_response.json(), {"ok": True, "conversation_complete": False, "session_ended": False})

        self.chat_session.completion_status = True
        self.chat_session.ended_at = timezone.now()
        self.chat_session.save(update_fields=["completion_status", "ended_at"])
        complete_response = self.post_json(
            "vip:student_voice_completion",
            {"voice_call_id": str(voice_call.pk)},
        )
        self.assertEqual(complete_response.json(), {"ok": True, "conversation_complete": True, "session_ended": True})


class SpeechEngineTranscriptGateTests(TransactionTestCase):
    reset_sequences = True

    def test_finalized_transcript_is_not_processed_before_introduction_completes(self):
        user = get_user_model().objects.create_user("intro-adapter-learner", password="test")
        user.groups.add(Group.objects.create(name="Class: Intro adapter"))
        prompt = RolePrompt.objects.create(title="Intro scenario", content=scenario_text(), is_active=True)
        chat_session = views._create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
        VoiceConversation.objects.create(
            chat_session=chat_session,
            provider_conversation_id="conv_intro",
        )
        speech_session = SimpleNamespace(
            conversation_id="conv_intro",
            send_response=AsyncMock(),
        )

        async_to_sync(SpeechEngineAdapter().on_transcript)(
            [SimpleNamespace(role="user", content="This is narration-era audio")],
            speech_session,
        )

        self.assertFalse(chat_session.messages.exists())
        speech_session.send_response.assert_not_awaited()

    def test_finalized_transcript_is_not_processed_while_muted(self):
        user = get_user_model().objects.create_user("muted-adapter-learner", password="test")
        user.groups.add(Group.objects.create(name="Class: Muted adapter"))
        prompt = RolePrompt.objects.create(title="Muted scenario", content=scenario_text(), is_active=True)
        chat_session = views._create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
        VoiceConversation.objects.create(
            chat_session=chat_session,
            provider_conversation_id="conv_muted",
            is_muted=True,
            introduction_completed_at=timezone.now(),
        )
        speech_session = SimpleNamespace(
            conversation_id="conv_muted",
            send_response=AsyncMock(),
        )

        async_to_sync(SpeechEngineAdapter().on_transcript)(
            [SimpleNamespace(role="user", content="This must remain muted")],
            speech_session,
        )

        self.assertFalse(chat_session.messages.exists())
        speech_session.send_response.assert_not_awaited()

    @patch(
        "vip.speech_engine.adapter._generate_and_persist_voice_response",
        return_value="[calm] First roleplay response.",
    )
    def test_first_finalized_transcript_after_introduction_uses_the_existing_turn_flow(self, generate_response):
        user = get_user_model().objects.create_user("ready-adapter-learner", password="test")
        user.groups.add(Group.objects.create(name="Class: Ready adapter"))
        prompt = RolePrompt.objects.create(title="Ready scenario", content=scenario_text(), is_active=True)
        chat_session = views._create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
        VoiceConversation.objects.create(
            chat_session=chat_session,
            provider_conversation_id="conv_ready",
            introduction_completed_at=timezone.now(),
        )
        speech_session = SimpleNamespace(
            conversation_id="conv_ready",
            send_response=AsyncMock(),
        )

        async_to_sync(SpeechEngineAdapter().on_transcript)(
            [SimpleNamespace(role="user", content="My first real response")],
            speech_session,
        )

        generate_response.assert_called_once()
        self.assertEqual(chat_session.messages.filter(sender=ChatMessage.Sender.STUDENT).count(), 1)
        speech_session.send_response.assert_awaited_once_with("[calm] First roleplay response.")


class PromptVoiceConfigurationTests(SimpleTestCase):
    @patch.dict("os.environ", {"ELEVENLABS_API_KEY": "server-secret"}, clear=False)
    @patch("elevenlabs.ElevenLabs")
    def test_voice_options_keep_provider_preview_urls(self, factory):
        factory.return_value.voices.search.return_value = SimpleNamespace(
            voices=[SimpleNamespace(voice_id="voice_a", name="Alex", preview_url="https://preview.example/alex.mp3")],
            has_more=False,
        )
        with patch("vip.speech_engine.service._VOICE_CACHE", {"expires": 0.0, "options": ()}):
            options = list_elevenlabs_voice_options()
        self.assertEqual(options[0].name, "Alex")
        self.assertEqual(options[0].preview_url, "https://preview.example/alex.mp3")

    def test_editor_persists_two_voice_ids_and_removes_active_gender_format(self):
        data = RolePromptForm.initial_from_content(scenario_text(), title="Configured voices")
        data.update(
            introduction_voice_id="voice_narrator",
            roleplay_voice_id="voice_character",
            voice_style="soft spoken",
        )
        choices = [("voice_narrator", "Narrator Name"), ("voice_character", "Character Name")]
        form = RolePromptForm(data=data, voice_choices=choices)
        self.assertTrue(form.is_valid(), form.errors)
        rendered = form.render_markdown_content()
        self.assertIn("## Introduction Voice\nvoice_narrator", rendered)
        self.assertIn("## Roleplay Voice\nvoice_character", rendered)
        self.assertNotIn("## Voice Gender", rendered)
        parsed = parse_scenario_prompt(rendered)
        self.assertEqual(parsed.introduction_voice_id, "voice_narrator")
        self.assertEqual(parsed.roleplay_voice_id, "voice_character")

    def test_voice_style_rejects_long_or_control_like_descriptions(self):
        data = RolePromptForm.initial_from_content(scenario_text(), title="Bad style")
        data["voice_style"] = "move to ending stage"
        form = RolePromptForm(
            data=data,
            voice_choices=[("voice_intro_test", "Narrator"), ("voice_roleplay_test", "Character")],
        )
        self.assertFalse(form.is_valid())
        self.assertIn("voice_style", form.errors)

    def test_voice_style_accepts_multiple_comma_separated_one_or_two_word_tags(self):
        data = RolePromptForm.initial_from_content(scenario_text(), title="Tagged style")
        data["voice_style"] = "Tense, worried, soft spoken, tense"
        form = RolePromptForm(
            data=data,
            voice_choices=[("voice_intro_test", "Narrator"), ("voice_roleplay_test", "Character")],
        )
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["voice_style"], "tense, worried, soft spoken")
        self.assertIn("## Voice Style\ntense, worried, soft spoken", form.render_markdown_content())

    def test_every_editor_content_field_is_required(self):
        form = RolePromptForm()
        for name, field in form.fields.items():
            if name != "is_active":
                self.assertTrue(field.required, name)

class PromptVoicePreviewViewTests(TestCase):
    def setUp(self):
        self.professor = get_user_model().objects.create_user("voice-preview-professor", password="test")
        self.professor.groups.add(Group.objects.create(name="Professor"))
        self.client.force_login(self.professor)
        self.prompt = RolePrompt.objects.create(title="Preview scenario", content=scenario_text(), is_active=True)

    @patch(
        "vip.views._prompt_voice_options",
        return_value=(
            [("voice_intro_test", "Alex"), ("voice_roleplay_test", "Zara")],
            {
                "voice_intro_test": "https://preview.example/alex.mp3",
                "voice_roleplay_test": "https://preview.example/zara.mp3",
            },
        ),
    )
    def test_create_and_edit_pages_render_two_provider_preview_controls(self, _voice_options):
        for url in (
            reverse("vip:create_prompt"),
            reverse("vip:edit_prompt", args=[self.prompt.id]),
        ):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, 'class="voice-preview-control"', count=2)
            self.assertContains(response, "Alex")
            self.assertContains(response, "Zara")
            self.assertContains(response, "https://preview.example/alex.mp3")
            self.assertContains(response, "https://preview.example/zara.mp3")
            self.assertNotContains(response, reverse("vip:student_voice_token"))

    @patch(
        "vip.views._prompt_voice_options",
        return_value=(
            [("voice_intro_saved", "Saved Narrator"), ("voice_roleplay_saved", "Saved Character")],
            {},
        ),
    )
    def test_edit_saves_voice_ids_reopens_selected_and_confirms_on_dashboard(self, _voice_options):
        data = RolePromptForm.initial_from_prompt(self.prompt)
        data.update(
            introduction_voice_id="voice_intro_saved",
            roleplay_voice_id="voice_roleplay_saved",
            voice_style="tense, soft spoken",
            is_active="on",
        )

        response = self.client.post(
            reverse("vip:edit_prompt", args=[self.prompt.id]),
            data,
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your prompt has been saved")
        self.prompt.refresh_from_db()
        scenario = parse_scenario_prompt(self.prompt.content)
        self.assertEqual(scenario.introduction_voice_id, "voice_intro_saved")
        self.assertEqual(scenario.roleplay_voice_id, "voice_roleplay_saved")
        self.assertEqual(scenario.voice_style, "tense, soft spoken")
        self.assertTrue(self.prompt.is_active)

        reopened = self.client.get(reverse("vip:edit_prompt", args=[self.prompt.id]))
        self.assertEqual(reopened.context["form"]["introduction_voice_id"].value(), "voice_intro_saved")
        self.assertEqual(reopened.context["form"]["roleplay_voice_id"].value(), "voice_roleplay_saved")
        self.assertContains(reopened, 'value="voice_intro_saved" selected')
        self.assertContains(reopened, 'value="voice_roleplay_saved" selected')

    def test_incomplete_prompt_is_deactivated_and_has_no_test_or_activate_action(self):
        incomplete_content = scenario_text().replace(
            "## Roleplay Voice\nvoice_roleplay_test\n",
            "## Roleplay Voice\n\n",
        )
        incomplete = RolePrompt.objects.create(
            title="Incomplete scenario",
            content=incomplete_content,
            is_active=True,
        )
        self.assertFalse(incomplete.is_active)
        self.assertFalse(incomplete.is_complete)

        dashboard = self.client.get(reverse("vip:professor_dashboard"), {"tab": "prompts"})
        self.assertContains(dashboard, "Incomplete")
        self.assertNotContains(
            dashboard,
            reverse("vip:professor_test_chat") + f"?prompt={incomplete.id}",
        )

        activation = self.client.post(
            reverse("vip:set_active_prompt", args=[incomplete.id]),
            follow=True,
        )
        self.assertContains(activation, "Complete every required prompt field")
        incomplete.refresh_from_db()
        self.assertFalse(incomplete.is_active)

        test_chat = self.client.get(reverse("vip:professor_test_chat"), {"prompt": incomplete.id})
        self.assertIsNone(test_chat.context["selected_prompt"])

    @patch(
        "vip.views._prompt_voice_options",
        return_value=(
            [("voice_intro_test", "Narrator"), ("voice_roleplay_test", "Character")],
            {},
        ),
    )
    def test_invalid_edit_deactivates_an_existing_active_prompt(self, _voice_options):
        self.prompt.is_active = True
        self.prompt.save(update_fields=["is_active", "updated_at"])
        original_content = self.prompt.content
        data = RolePromptForm.initial_from_prompt(self.prompt)
        data.update(is_active="on", closing="")

        response = self.client.post(
            reverse("vip:edit_prompt", args=[self.prompt.id]),
            data,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The prompt was not saved")
        self.prompt.refresh_from_db()
        self.assertFalse(self.prompt.is_active)
        self.assertEqual(self.prompt.content, original_content)


class SpeechEngineVoiceResourceTests(TestCase):
    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_base"},
        clear=False,
    )
    @patch("elevenlabs.ElevenLabs")
    def test_selected_roleplay_voice_provisions_and_reuses_fixed_voice_resource(self, factory):
        client = factory.return_value
        client.speech_engine.get.return_value = SimpleNamespace(
            config=SimpleNamespace(speech_engine={"ws_url": "wss://voice.example/ws"})
        )
        client.speech_engine.create.return_value = SimpleNamespace(engine_id="seng_selected_voice")
        client.conversational_ai.conversations.get_webrtc_token.return_value = SimpleNamespace(
            token="short-lived-token",
            conversation_id="conv_selected_voice",
        )

        token = issue_webrtc_token("learner", voice_id="voice_character")
        create_kwargs = client.speech_engine.create.call_args.kwargs
        self.assertEqual(create_kwargs["tts"]["voice_id"], "voice_character")
        self.assertEqual(create_kwargs["tts"]["model_id"], "eleven_v3_conversational")
        self.assertTrue(create_kwargs["tts"]["expressive_mode"])
        self.assertEqual(create_kwargs["overrides"], {"first_message": False})
        self.assertEqual(create_kwargs["turn"], {"silence_end_call_timeout": -1})
        self.assertEqual(create_kwargs["conversation"]["max_duration_seconds"], 600)
        self.assertIn("agent_response_complete", create_kwargs["conversation"]["client_events"])
        self.assertEqual(
            SpeechEngineVoiceResource.objects.get(voice_id="voice_character").speech_engine_id,
            "seng_selected_voice",
        )

        issue_webrtc_token("learner", voice_id="voice_character")
        self.assertEqual(client.speech_engine.create.call_count, 1)
        self.assertEqual(
            client.conversational_ai.conversations.get_webrtc_token.call_args.kwargs["agent_id"],
            "seng_selected_voice",
        )


class SpeechEngineAdapterTests(TestCase):
    def test_socket_close_diagnostics_include_provider_close_code_and_reason(self):
        code, reason = speech_engine_adapter._socket_close_details(
            SimpleNamespace(_ws=SimpleNamespace(close_code=1006, close_reason="network transport lost"))
        )
        self.assertEqual(code, 1006)
        self.assertEqual(reason, "network transport lost")

    def test_turn_id_distinguishes_repeated_spoken_text_by_finalized_turn_position(self):
        adapter = SpeechEngineAdapter()
        first = [SimpleNamespace(role="user", content="Same words")]
        second = [
            SimpleNamespace(role="user", content="Same words"),
            SimpleNamespace(role="agent", content="Reply"),
            SimpleNamespace(role="user", content="Same words"),
        ]
        self.assertNotEqual(adapter._turn_id("conv_123", first), adapter._turn_id("conv_123", second))
        self.assertEqual(adapter._latest_user_text(second), "Same words")

    @patch("vip.speech_engine.adapter._run_sync", new_callable=AsyncMock)
    def test_provider_error_logs_the_mapped_voice_and_chat_session_context(self, run_sync):
        run_sync.side_effect = [
            SimpleNamespace(pk="voice-call-id", chat_session_id=73),
            None,
        ]

        with self.assertLogs("vip.speech_engine.adapter", level="ERROR") as logs:
            async_to_sync(SpeechEngineAdapter().on_error)(
                RuntimeError("provider disconnected"),
                SimpleNamespace(conversation_id="conv_error_123"),
            )

        output = "\n".join(logs.output)
        self.assertIn("conv_error_123", output)
        self.assertIn("voice-call-id", output)
        self.assertIn("chat_session=73", output)
        self.assertEqual(run_sync.await_count, 2)

    def test_adapter_claims_and_persists_through_the_existing_text_turn_lifecycle(self):
        user = get_user_model().objects.create_user("adapter-learner", password="test")
        user.groups.add(Group.objects.create(name="Class: Adapter"))
        prompt = RolePrompt.objects.create(title="Adapter scenario", content=scenario_text(), is_active=True)
        chat_session = views._create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
        claim = speech_engine_adapter._claim_voice_turn(chat_session.id, "Tell me what happens next.", "eleven_test_1")
        self.assertEqual(claim.status, "accepted")
        with patch(
            "vip.speech_engine.adapter._generate_assistant_response",
            return_value=(
                "[sighs] A persisted reply. [pleading] Please stay with me.",
                False,
                {"voice_metadata": "calm"},
            ),
        ):
            reply = speech_engine_adapter._generate_and_persist_voice_response(claim, threading.Event())
        self.assertEqual(reply, "[sighs] A persisted reply. [pleading] Please stay with me.")
        self.assertEqual(chat_session.messages.filter(sender="student").count(), 1)
        self.assertEqual(
            chat_session.messages.filter(sender="assistant", turn_id="eleven_test_1").get().content,
            "[sighs] A persisted reply. [pleading] Please stay with me.",
        )

    def test_provider_disconnect_closes_the_durable_voice_session(self):
        user = get_user_model().objects.create_user("adapter-disconnect-learner", password="test")
        user.groups.add(Group.objects.create(name="Class: Adapter disconnect"))
        prompt = RolePrompt.objects.create(title="Disconnect scenario", content=scenario_text(), is_active=True)
        chat_session = views._create_chat_session(user, prompt, ChatSession.InteractionMode.VOICE)
        voice_call = VoiceConversation.objects.create(
            chat_session=chat_session,
            provider_conversation_id="conv_disconnected",
        )
        claimed, status, _ = views._accept_learner_turn(chat_session, "I am speaking", "disconnect-turn")
        self.assertEqual(status, "accepted")

        speech_engine_adapter._mark_voice_ended("conv_disconnected", failed=True)

        voice_call.refresh_from_db()
        chat_session.refresh_from_db()
        self.assertEqual(voice_call.status, VoiceConversation.Status.ERROR)
        self.assertEqual(voice_call.failure_reason, "Voice connection ended unexpectedly.")
        self.assertIsNotNone(chat_session.ended_at)
        self.assertEqual(chat_session.active_turn_id, "")
        self.assertFalse(
            views._persist_assistant_response(
                claimed,
                "late reply",
                False,
                {},
                "disconnect-turn",
                claim_id=claimed.active_claim_id,
            )
        )


class ProfessorTestChatModeTests(TestCase):
    def setUp(self):
        self.professor = get_user_model().objects.create_user("test-professor", password="test")
        self.professor.groups.add(Group.objects.create(name="Professor"))
        self.client.force_login(self.professor)
        self.prompt = RolePrompt.objects.create(
            title="Draft voice scenario",
            content=scenario_text(),
            is_active=False,
        )
        self.url = reverse("vip:professor_test_chat")

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    def test_test_chat_shows_mode_picker_for_draft_scenario(self):
        response = self.client.get(self.url, {"prompt": self.prompt.id})
        self.assertContains(response, "How would you like to practice?")
        self.assertContains(response, "Start Voice Conversation")
        self.assertContains(response, "Start Text Conversation")
        self.assertContains(response, "(draft)")
        self.assertContains(response, "voice_chat.js")
        self.assertNotContains(response, 'id="student-message-input"')

    def test_switching_prompts_from_new_chat_keeps_history_hidden(self):
        previous_prompt = RolePrompt.objects.create(
            title="Previously used scenario",
            content=scenario_text(),
            is_active=False,
        )
        previous_session = views._create_chat_session(
            self.professor,
            previous_prompt,
            ChatSession.InteractionMode.TEXT,
        )
        ChatMessage.objects.create(
            session=previous_session,
            sender=ChatMessage.Sender.STUDENT,
            content="old test-chat history sentinel",
        )

        prompt_page = self.client.get(self.url, {"prompt": previous_prompt.id})
        self.assertContains(prompt_page, "How would you like to practice?")
        self.assertIsNone(prompt_page.context["current_session"])
        self.assertNotContains(prompt_page, "old test-chat history sentinel")

        new_chat = self.client.get(self.url, {"prompt": self.prompt.id, "new": "1"})
        self.assertContains(new_chat, '<input type="hidden" name="new" value="1">')

        switched_prompt = self.client.get(
            self.url,
            {"prompt": previous_prompt.id, "new": "1"},
        )
        self.assertContains(switched_prompt, "How would you like to practice?")
        self.assertIsNone(switched_prompt.context["current_session"])
        self.assertNotContains(switched_prompt, "old test-chat history sentinel")

        opened_history = self.client.get(
            self.url,
            {"session": previous_session.id, "prompt": previous_prompt.id},
        )
        self.assertEqual(opened_history.context["current_session"].id, previous_session.id)
        self.assertContains(opened_history, "old test-chat history sentinel")

    def test_professor_text_mode_uses_text_only_composer(self):
        response = self.client.post(
            self.url + f"?prompt={self.prompt.id}&new=1",
            {"action": "start_session", "mode": "text", "prompt_id": self.prompt.id},
        )
        session = ChatSession.objects.latest("id")
        self.assertEqual(session.student, self.professor)
        self.assertEqual(session.interaction_mode, ChatSession.InteractionMode.TEXT)
        page = self.client.get(response.url)
        self.assertContains(page, "Text Conversation")
        self.assertContains(page, 'id="student-message-input"')
        self.assertNotContains(page, "voice_chat.js")
        self.assertNotContains(page, 'id="voice-mute"')

    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    @patch("vip.views.issue_webrtc_token", return_value=VoiceToken("browser-token", "conv_professor_123"))
    def test_professor_can_start_voice_on_draft_and_map_it_to_own_test_session(self, issue_token):
        response = self.client.post(
            self.url + f"?prompt={self.prompt.id}&new=1",
            {"action": "start_session", "mode": "voice", "prompt_id": self.prompt.id},
        )
        session = ChatSession.objects.latest("id")
        self.assertEqual(session.interaction_mode, ChatSession.InteractionMode.VOICE)
        self.assertEqual(session.student, self.professor)
        page = self.client.get(response.url)
        self.assertContains(page, "Start Call")
        self.assertNotContains(page, 'id="student-message-input"')

        token_response = self.client.post(
            reverse("vip:student_voice_token"),
            data=json.dumps({"prompt_id": self.prompt.id, "session_id": session.id}),
            content_type="application/json",
        )
        self.assertEqual(token_response.status_code, 200)
        data = token_response.json()
        self.assertEqual(data["chat_session_id"], session.id)
        self.assertEqual(VoiceConversation.objects.get(pk=data["voice_call_id"]).chat_session, session)
        issue_token.assert_called_once_with(
            participant_name=self.professor.username,
            voice_id="voice_roleplay_test",
        )

        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test"}, clear=False), patch(
            "vip.views.ElevenLabsSpeechStream", return_value=iter([b"narration"])
        ) as stream:
            introduction = self.client.get(
                reverse("vip:student_voice_introduction", args=[session.id])
            )
            self.assertTrue(introduction.streaming)
            self.assertEqual(list(introduction.streaming_content), [b"narration"])
        self.assertEqual(stream.call_args.kwargs["voice_id"], "voice_intro_test")

    def test_professor_voice_token_still_cannot_take_another_users_session(self):
        other = get_user_model().objects.create_user("other-professor-target")
        other.groups.add(Group.objects.get(name="Professor"))
        session = views._create_chat_session(other, self.prompt, ChatSession.InteractionMode.VOICE)
        with patch.dict(
            "os.environ",
            {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
            clear=False,
        ), patch("vip.views.issue_webrtc_token", return_value=VoiceToken("browser-token", "conv_new_professor")):
            response = self.client.post(
                reverse("vip:student_voice_token"),
                data=json.dumps({"prompt_id": self.prompt.id, "session_id": session.id}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()["chat_session_id"], session.id)


class SpeechEngineSdkWiringTests(TestCase):
    @patch.dict(
        "os.environ",
        {"ELEVENLABS_API_KEY": "server-secret", "ELEVENLABS_SPEECH_ENGINE_ID": "seng_test"},
        clear=False,
    )
    @patch("elevenlabs.AsyncElevenLabs")
    def test_adapter_uses_sdk_server_with_provider_auth_left_enabled(self, factory):
        client = factory.return_value
        engine = MagicMock()
        engine.serve = AsyncMock()
        client.speech_engine.get = AsyncMock(return_value=engine)
        async_to_sync(serve_speech_engine)(port=3011, debug=False)
        engine.serve.assert_awaited_once()
        kwargs = engine.serve.call_args.kwargs
        self.assertEqual(kwargs["port"], 3011)
        self.assertEqual(kwargs["path"], "/ws")
        self.assertNotIn("disable_auth", kwargs)
        self.assertIn("on_transcript", kwargs)


class VoiceBrowserAssetContractTests(SimpleTestCase):
    def test_speech_engine_connects_only_after_narration_without_a_first_message_override(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertLess(
            source.index("await playNarrator(tokenData.introduction_audio_url)"),
            source.index("await connectAfterIntroduction(Conversation, tokenData.token)"),
        )
        self.assertNotIn('firstMessage: ""', source)
        self.assertNotIn("firstMessage:", source)
        self.assertIn("await conversation.setMicMuted(true)", source)
        self.assertIn("await postJSON(root.dataset.readyUrl", source)
        self.assertIn("await finalizeAndShowTranscript({failed: true})", source)
        self.assertIn('setStatus("Simulation introduction...")', source)

    def test_only_mute_remains_as_an_audio_input_control(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("async function toggleMute", source)
        self.assertIn("await conversation.setMicMuted(keepMicrophoneMuted())", source)
        self.assertIn("root.dataset.muteUrl", source)
        self.assertIn("await postJSON(root.dataset.muteUrl", source)
        self.assertIn("&& !muted", source)
        self.assertNotIn("pauseVoice", source)
        self.assertNotIn("resumeVoice", source)
        self.assertNotIn("pauseUrl", source)

    def test_completed_or_failed_calls_open_the_saved_transcript_instead_of_retrying(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("async function finalizeAndShowTranscript", source)
        self.assertIn("window.location.assign(transcriptUrl)", source)
        self.assertIn("finalizeAndShowTranscript({failed: true})", source)
        self.assertNotIn("Retry to continue this saved session.", source)

    def test_intentional_end_finalizes_before_closing_the_provider_connection(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        end_start = source.index("async function endVoice()")
        end_body = source[end_start:source.index("choice?.closest", end_start)]
        self.assertLess(end_body.index("await finalizeVoiceCall()"), end_body.index("conversation.endSession()"))

    def test_live_interim_and_final_transcripts_share_a_deduplicating_finalizer(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("onTentativeUserTranscript", source)
        self.assertIn("onUserTranscript", source)
        self.assertIn("function finalizeUserTranscript", source)
        self.assertIn("renderedEvents.has(eventKey)", source)
        self.assertIn("now - recent.at < 2000", source)
        self.assertIn("function cleanProviderAudioTags", source)

    def test_unexpected_disconnect_has_a_distinct_recovery_state_and_diagnostic(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn('"Connection lost"', source)
        self.assertIn("async function handleUnexpectedConnectionLoss", source)
        self.assertIn("function reportConnectionDiagnostic", source)
        self.assertIn("connectionLost.hidden = false", source)
        self.assertIn("voice-open-saved-transcript", source)
        self.assertIn("if (endedByUser)", source)
        self.assertIn("root.dataset.diagnosticUrl", source)

    def test_completed_conversation_waits_for_post_audio_event_before_ending(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("agent_response_complete", source)
        self.assertIn("async function endCompletedConversationAfterAudio", source)
        self.assertIn("async function inspectCompletedConversation", source)
        self.assertIn("root.dataset.completionUrl", source)
        debug_start = source.index("function handleProviderDebugEvent")
        debug_body = source[debug_start:source.index("async function finalizeVoiceCall", debug_start)]
        self.assertNotIn("markAgentAudioComplete", debug_body)
        self.assertIn("if (mode === \"listening\") markAgentAudioComplete()", source)
        automatic_start = source.index("async function endCompletedConversationAfterAudio")
        automatic_body = source[automatic_start:source.index("function markAgentAudioComplete", automatic_start)]
        self.assertLess(automatic_body.index("await conversation.endSession()"), automatic_body.index("await finalizeAndShowTranscript()"))

    def test_visible_transcript_uses_a_generic_inline_audio_tag_parser(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("const inlineAudioTagPattern", source)
        self.assertIn("[a-z][a-z'-]*", source)
        self.assertIn("function cleanProviderAudioTags", source)
        self.assertNotIn('"clears throat"', source)

    def test_intro_input_is_gated_until_the_narrator_completion_handoff(self):
        source = (Path(settings.BASE_DIR) / "vip" / "static" / "vip" / "voice_chat.js").read_text(encoding="utf-8")
        self.assertIn("function canProcessUserSpeech", source)
        self.assertIn("if (!canProcessUserSpeech()) return;", source)
        self.assertIn("async function completeIntroduction", source)
        self.assertIn("root.dataset.readyUrl", source)
        self.assertIn("await conversation.setMicMuted(keepMicrophoneMuted())", source)
        self.assertIn("transitionToListeningOnce", source)
        self.assertLess(
            source.index("await playNarrator(tokenData.introduction_audio_url)"),
            source.index("await connectAfterIntroduction(Conversation, tokenData.token)"),
        )

    def test_prompt_editor_plays_provider_samples_without_starting_a_voice_session(self):
        source = (Path(settings.BASE_DIR) / "vip" / "templates" / "vip" / "prompt_form.html").read_text(encoding="utf-8")
        self.assertIn("voice-preview-control", source)
        self.assertIn("voice_preview_urls|json_script", source)
        self.assertIn("new Audio(previewUrl)", source)
        self.assertIn("function stopVoicePreview", source)
        self.assertNotIn("student_voice_token", source)
