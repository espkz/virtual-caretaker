import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from vip import clinician_demo, core_questions, views
from vip.conversation_engine import ConversationEngine
from vip.conversation_graph import MAX_TURNS
from vip.conversation_scenario import Scenario, parse_scenario_prompt
from vip.forms import RolePromptForm
from vip.models import RolePrompt
from . import test_practice as practice
from .test_practice import message


def nurse_text():
    return (Path(settings.BASE_DIR).parent / "prompts" / "role_hospice_nurse_2.md").read_text(encoding="utf-8")


def reply(text="The PEG stays in place for prescribed comfort medications.", topics=(), intent="continue", readiness=False, check="grounded", kind="answer"):
    return {"dialogue": text, "addressed_topics": list(topics), "user_intent": intent,
            "readiness_check": readiness, "self_check": check, "reply_kind": kind}


class ClinicianEngineTests(SimpleTestCase):
    def setUp(self):
        self.text = nurse_text()
        self.scenario = parse_scenario_prompt(self.text)
        self.ids = [objective.id for objective in self.scenario.objectives]
        self.engine = ConversationEngine("Never sound like a clinician; patient-mode instructions.", "test")

    def respond(self, result, learner="Won't she feel hungry?", state=None):
        with patch.object(self.engine, "_request_clinician_reply", return_value=result):
            return self.engine.respond(self.text, [message("student", learner)], state)

    def test_explicit_mode_round_trips_without_entering_family_question_path(self):
        self.assertEqual(self.scenario.simulation_mode, "clinician_demo")
        self.assertFalse(core_questions.enabled(self.scenario))
        self.assertEqual(Scenario.from_state(self.scenario.to_state()), self.scenario)

    def test_first_question_is_answered_and_legitimate_nurse_speech_is_not_blocked(self):
        answer = "The PEG stays in place for prescribed comfort medications. We will continue comfort care."
        with patch.object(self.engine, "_assess_core_reply") as family:
            text, complete, state = self.respond(reply(answer, topics=[self.ids[1]]))
        self.assertEqual(text, answer)
        self.assertFalse(complete)
        self.assertEqual(state["covered_topics"], [self.ids[1]])
        family.assert_not_called()

    def test_family_opening_cannot_override_first_clinician_reply(self):
        role = self.text + "\n## Opening Line\nI am Rachel and I am worried.\n"
        with patch.object(self.engine, "_request_clinician_reply", return_value=reply("I hear your concern.")):
            answer, _, _ = self.engine.respond(role, [message("student", "Please explain.")])
        self.assertEqual(answer, "I hear your concern.")

    def test_closing_requires_prior_readiness_check_and_all_topics(self):
        for state in ({}, {"ending_ready": True, "covered_topics": self.ids[:1]}):
            _, complete, _ = self.respond(reply(intent="ready"), learner="I'm ready.", state=state)
            self.assertFalse(complete)
        _, complete, state = self.respond(reply(topics=self.ids, readiness=True))
        self.assertFalse(complete)
        text, complete, _ = self.respond(reply(intent="ready"), learner="I'm ready to proceed.", state=state)
        self.assertTrue(complete)
        self.assertEqual(text, self.scenario.closing)
        self.assertNotIn("I'm just a mess", text)

    def test_thanks_and_outstanding_question_do_not_close(self):
        state = {"ending_ready": True, "covered_topics": self.ids}
        _, complete, _ = self.respond(reply(), learner="Thank you.", state=state)
        self.assertFalse(complete)
        _, complete, _ = self.respond(reply(intent="ready"), learner="Thank you.", state=state)
        self.assertFalse(complete)
        _, complete, _ = self.respond(reply(intent="ready"), learner="I'm ready, but will she suffer?", state=state)
        self.assertFalse(complete)

    def test_role_and_dose_boundaries_do_not_advance_progress(self):
        for text, reason in [
            ("Rachel: I am afraid. Nurse: Let me help.", "clinician_role_boundary"),
            ("I'm Rachel. My mother needs help.", "clinician_role_boundary"),
            ("Give 20 mg now.", "clinician_dose_boundary"),
            ("Give five milliliters.", "clinician_dose_boundary"),
            ("Give two tablets every four hours.", "clinician_dose_boundary"),
            ("Give it every hour.", "clinician_dose_boundary"),
        ]:
            with self.subTest(text=text):
                answer, complete, state = self.respond(reply(text, topics=self.ids, intent="ready", readiness=True))
                self.assertNotEqual(answer, text)
                self.assertFalse(complete)
                self.assertEqual(state["reason"], reason)
                self.assertEqual(state["covered_topics"], [])

    def test_unsupported_fact_uses_check_with_team_response(self):
        answer, _, state = self.respond(reply("I checked her latest scan this morning.", topics=self.ids, check="unsupported"))
        self.assertEqual(answer, clinician_demo.SCOPE_REDIRECT)
        self.assertEqual(state["covered_topics"], [])

    def test_coverage_is_monotonic_and_unknown_ids_are_ignored(self):
        _, _, state = self.respond(reply(topics=[self.ids[2], "invented"]), state={"covered_topics": [self.ids[0]]})
        self.assertEqual(state["covered_topics"], [self.ids[0], self.ids[2]])

    def test_stopping_chat_is_distinct_from_stopping_feeding(self):
        with patch.object(self.engine, "_request_clinician_reply") as request:
            text, complete, _ = self.engine.respond(self.text, [message("student", "Please stop.")])
            request.assert_not_called()
        self.assertTrue(complete)
        self.assertEqual(text, clinician_demo.PAUSE_CLOSING)
        _, complete, _ = self.respond(reply(), learner="Why stop the feeding pump?")
        self.assertFalse(complete)

    def test_hard_limit_has_a_clinician_closing_without_api(self):
        with patch.object(self.engine, "_request_clinician_reply") as request:
            text, complete, _ = self.engine.respond(self.text, [message("student", "I'm uncertain.")] * MAX_TURNS)
        request.assert_not_called()
        self.assertTrue(complete)
        self.assertEqual(text, clinician_demo.LIMIT_CLOSING)

    def test_pause_does_not_claim_readiness_for_treatment(self):
        _, complete, state = self.respond(reply(intent="pause"), learner="I need a break from this conversation.")
        self.assertTrue(complete)
        self.assertFalse(state["ending_ready"])

    def test_malformed_response_is_retryable(self):
        for result in ({}, reply(text=""), {**reply(), "addressed_topics": "all"}):
            with self.assertRaises(ValueError):
                self.respond(result)

    def test_api_payload_excludes_family_instructions_and_labels_human_as_user(self):
        client = MagicMock()
        client.responses.create.return_value.output_text = json.dumps(reply())
        with patch.object(self.engine, "_openai_client", return_value=client):
            self.engine.respond(self.text, [message("student", "My brother is giving up.")])
        request = client.responses.create.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertEqual(request["input"][-1]["role"], "user")
        system = request["input"][0]["content"]
        self.assertNotIn("Never sound like a clinician", system)
        self.assertIn("Rachel", system)
        self.assertIn("four months", system)

    def test_editor_preserves_mode_and_clinical_guidance(self):
        data = RolePromptForm.initial_from_content(self.text, title="Nurse demo")
        form = RolePromptForm(data=data)
        self.assertTrue(form.is_valid(), form.errors)
        scenario = parse_scenario_prompt(form.render_markdown_content())
        self.assertEqual(scenario.simulation_mode, "clinician_demo")
        self.assertEqual(scenario.background_context, self.scenario.background_context)
        self.assertEqual(scenario.middle, self.scenario.middle)
        self.assertEqual(scenario.learner, self.scenario.learner)


class ClinicianWorkflowTests(TestCase):
    post = practice.ChatWorkflowTests.post

    def setUp(self):
        practice.ChatWorkflowTests.setUp(self)
        self.prompt.content = nurse_text()
        self.prompt.save()
        self.session = views._create_chat_session(self.user, self.prompt)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_nurse_reply_and_readiness_survive_browser_requests(self):
        ids = [objective.id for objective in parse_scenario_prompt(nurse_text()).objectives]
        with patch.object(ConversationEngine, "_request_clinician_reply", return_value=reply(topics=ids, readiness=True)):
            self.assertEqual(self.post(text="I have questions about all three concerns.").status_code, 302)
        self.session.refresh_from_db()
        self.assertEqual(self.session.covered_topics, ids)
        self.assertTrue(self.session.ending_ready)
        with patch.object(ConversationEngine, "_request_clinician_reply", return_value=reply(intent="ready")):
            self.post(text="I understand now and feel ready.", turn="turn2")
        self.session.refresh_from_db()
        self.assertTrue(self.session.completion_status)
        self.assertIsNotNone(self.session.ended_at)

    def test_opt_in_import_creates_separate_draft_and_preserves_existing_prompts(self):
        original = self.prompt.content
        for _ in range(2):
            call_command("load_scenarios", clinician_demo=True, stdout=io.StringIO())
        imported = RolePrompt.objects.get(title="Scenario 2: AI hospice nurse (you play Rachel)")
        self.assertFalse(imported.is_active)
        self.assertEqual(imported.simulation_mode_label, "Clinician demonstration")
        self.prompt.refresh_from_db()
        self.assertEqual(self.prompt.content, original)

    def test_editor_displays_mode_selector(self):
        from django.contrib.auth.models import Group
        self.user.groups.add(Group.objects.create(name="Professor"))
        response = self.client.get(reverse("vip:edit_prompt", args=[self.prompt.id]))
        self.assertContains(response, 'value="clinician_demo" selected')
