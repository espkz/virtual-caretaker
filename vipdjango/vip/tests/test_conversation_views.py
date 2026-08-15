import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vipson_manager.settings")

import django

django.setup()

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse

from vip.models import ChatMessage, ChatSession, RolePrompt
from vip.views import (
    _conversation_messages,
    _rendered_chat_messages,
    split_dialogue_and_voice,
    student_message_tts,
)


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
        self.assertEqual(rendered[1]["display_content"], "I said [this] to Rachel.")

        history = _conversation_messages([assistant, learner])
        self.assertEqual(history[0], {"role": "assistant", "content": "I am scared."})
        self.assertEqual(history[1], {"role": "user", "content": "I said [this] to Rachel."})
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
