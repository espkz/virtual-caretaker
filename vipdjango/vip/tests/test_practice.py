import io
import json
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from vip import core_questions, views
from vip.conversation_engine import ConversationEngine, clean_dialogue_for_display, split_dialogue_and_voice
from vip.conversation_graph import MAX_TURNS
from vip.conversation_scenario import parse_scenario_prompt
from vip.models import ChatMessage, ChatSession, RolePrompt
from vip.speech import SpeechStream


def scenario_text(number=1):
    content = (Path(settings.BASE_DIR).parent / "prompts" / f"role_rachel_ellison_{number}.md").read_text(encoding="utf-8")
    content = re.sub(
        r"(?m)^(## Introduction Voice\n)[^\n]*$",
        r"\1voice_intro_test",
        content,
    )
    content = re.sub(
        r"(?m)^(## Roleplay Voice\n)[^\n]*$",
        r"\1voice_roleplay_test",
        content,
    )
    content = re.sub(
        r"(?m)^(## Voice Style\n)[^\n]*$",
        r"\1worried, hesitant",
        content,
    )
    return "## Simulation Mode\nroleplay\n\n" + content


def message(sender, content):
    return SimpleNamespace(sender=sender, content=content)


def selection(status="addressed", question="", dialogue="That is difficult to take in.", **extra):
    return {
        "answer_status": status,
        "question_id": question,
        "dialogue": dialogue,
        "addressed_question_ids": [],
        "ready_to_close": False,
        "readiness_evidence": "",
        **extra,
    }


def scripted_reply(scenario, messages, pending, available, progress):
    question = available[0] if available else None
    return selection(
        question=question["id"] if question else "",
        dialogue=(
            question["text"]
            if question
            else f"I need a moment to think about what you said at this point ({progress['turn']})."
        ),
    )


class CorePracticeTests(SimpleTestCase):
    def run_scenario(self, number, status="addressed"):
        engine = ConversationEngine("", "test")
        transcript, state, outputs = [], {}, []

        def reply(*args):
            result = scripted_reply(*args)
            result["answer_status"] = status
            return result

        with patch.object(engine, "_assess_core_reply", side_effect=reply) as model:
            for turn in range(1, MAX_TURNS + 1):
                transcript.append(message("student", "Here is my response."))
                text, complete, state = engine.respond(scenario_text(number), transcript, state)
                transcript.append(message("assistant", text))
                outputs.append(text)
                if complete:
                    break
        return turn, state, outputs, model

    def test_both_scenarios_allow_twenty_five_exchanges_and_all_themes(self):
        for number in (1, 2):
            turn, state, outputs, model = self.run_scenario(number)
            self.assertEqual(turn, 25)
            self.assertEqual(model.call_count, 25)
            self.assertTrue(state["completion_status"])
            self.assertEqual(state["reason"], "turn_limit")
            self.assertGreater(len(state["core_question_state"]["asked"]), 6)
            self.assertTrue(all(count >= 3 for count in state["topic_turn_counts"].values()))
            self.assertNotIn("?", outputs[-1])

    def test_unclear_and_unsafe_answers_do_not_loop_or_falsely_succeed(self):
        for status in ("unclear", "unsafe"):
            for number in (1, 2):
                turn, state, outputs, _ = self.run_scenario(number, status)
                self.assertEqual(turn, 25)
                self.assertNotIn("explain that concern in simpler terms", " ".join(outputs))
                self.assertNotIn("speak with someone", " ".join(outputs))
                self.assertNotEqual(outputs[-1], parse_scenario_prompt(scenario_text(number)).closing)
                self.assertFalse(state["ending_ready"])

    def test_nurse_question_is_answered_without_losing_pending_concern(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        answer = "I know her heart stopped and her brain went without oxygen. But her eyes are open."
        text, complete, state = core_questions.respond(
            scenario,
            [message("student", "What do you understand?")],
            progress,
            lambda *args: selection("learner_question", dialogue=answer),
        )
        self.assertEqual(text, answer)
        self.assertFalse(complete)
        self.assertEqual(state["core_question_state"]["pending"]["id"], "opening")
        self.assertEqual(state["core_question_state"]["addressed"], [])

    def test_stop_works_before_opening_without_api(self):
        engine = ConversationEngine("", "test")
        with patch.object(engine, "_assess_core_reply") as model:
            text, complete, state = engine.respond(scenario_text(), [message("student", "Please stop.")])
        model.assert_not_called()
        self.assertTrue(complete)
        self.assertEqual(state["reason"], "learner_stop")

    def test_topic_eligible_roleplay_defaults_to_core_question_path(self):
        engine = ConversationEngine("", "test")
        transcript = [message("student", "Hello")]
        with patch.object(engine, "_assess_core_reply", side_effect=scripted_reply) as core_model:
            with patch.object(engine, "_request_llm_turn") as general_model:
                _, complete, state = engine.respond(scenario_text(), transcript)

        self.assertFalse(complete)
        core_model.assert_called_once()
        general_model.assert_not_called()
        self.assertEqual(state["reason"], "scenario_dialogue")

    def test_roleplay_without_parsed_topics_uses_general_llm_graph(self):
        role_text = """
## Simulation Mode
roleplay

## Role
You are a character who is worried but willing to talk.

## Background and Context
The character has a personal concern.

## User Role
The human participant is a nurse.

## Introduction
Welcome to the simulation.

## Conversation Stages

### Beginning
Respond naturally to the nurse's latest message.

### Middle
Continue the conversation based on unresolved concerns.

### Ending
Wrap up when the conversation reaches a natural endpoint.

## Closing
Thank you. I am ready to continue.
""".strip()
        engine = ConversationEngine("", "test")
        result = {
            "dialogue": "[worried] I am still concerned.",
            "stage": "beginning",
            "stage_transition_ready": False,
            "complete": False,
            "stop_requested": False,
            "active_objective": "",
            "covered_objectives": [],
            "unresolved_objectives": [],
            "ending_ready": False,
            "active_topic": "",
            "covered_topics": [],
            "unresolved_topics": [],
        }
        with patch.object(engine, "_request_llm_turn", return_value=result) as general_model:
            with patch.object(engine, "_assess_core_reply") as core_model:
                _, complete, state = engine.respond(role_text, [message("student", "Hello")])

        self.assertFalse(complete)
        general_model.assert_called_once()
        core_model.assert_not_called()
        self.assertEqual(state["reason"], "llm_turn")

    def test_twenty_fifth_input_gets_response_then_closes(self):
        engine = ConversationEngine("", "test")
        with patch.object(
            engine,
            "_assess_core_reply",
            return_value=selection(dialogue="I understand that you will stay with us. What happens next?"),
        ) as model:
            text, complete, state = engine.respond(
                scenario_text(),
                [message("student", "I will stay with you.")] * MAX_TURNS,
            )
        model.assert_called_once()
        self.assertIn("I understand that you will stay with us.", text)
        self.assertNotIn("?", text)
        self.assertTrue(complete)
        self.assertEqual(state["reason"], "turn_limit")

    def test_legacy_cap_and_completed_session_do_not_call_model(self):
        engine = ConversationEngine("", "test")
        _, complete, state = engine.respond(
            "## Role\nYou are a worried patient.",
            [message("student", "hello")] * MAX_TURNS,
        )
        self.assertTrue(complete)
        self.assertEqual(state["reason"], "turn_limit")
        self.assertEqual(engine.respond(scenario_text(), [], {"completion_status": True})[:2], ("", True))

    def test_later_repair_clears_earlier_unresolved_concern(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        progress["unresolved"] = ["opening"]
        _, _, state = core_questions.respond(
            scenario,
            [],
            progress,
            lambda *args: selection(question=args[3][0]["id"], dialogue=args[3][0]["text"]),
        )
        self.assertEqual(state["core_question_state"]["unresolved"], [])

    def test_invalid_dialogue_or_ids_fail_without_mutating_committed_state(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        before = json.dumps(progress)
        for result in (
            {},
            selection(question="invented"),
            selection(dialogue="I am your nurse. Give her 100 mg."),
            selection(dialogue="I'd like to speak with someone from the care team."),
        ):
            with self.assertRaises(ValueError):
                core_questions.respond(scenario, [], progress, lambda *args: result)
            self.assertEqual(before, json.dumps(progress))

    def test_model_request_includes_full_conditional_guidance_pacing_and_audio_tags(self):
        engine = ConversationEngine("", "test")
        client = MagicMock()
        client.responses.create.return_value.output_text = json.dumps(selection())
        scenario = parse_scenario_prompt(scenario_text(2))
        with patch.object(engine, "_openai_client", return_value=client):
            engine._assess_core_reply(
                scenario,
                [message("student", "What do you know?")],
                None,
                [],
                {"turn": 2},
            )
        request = client.responses.create.call_args.kwargs
        context = request["input"][0]["content"]
        self.assertFalse(request["store"])
        self.assertEqual(request["input"][-1]["role"], "user")
        for fragment in (
            "What Rachel knows before the nurse arrives",
            "How Rachel responds",
            "learner_question",
            "25 learner",
            "readiness",
            "ElevenLabs v3 audio tag",
        ):
            self.assertIn(fragment, context)
        self.assertIn("dialogue", request["text"]["format"]["schema"]["properties"])

    def test_unclear_answer_allows_one_specific_followup(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        _, complete, state = core_questions.respond(
            scenario,
            [],
            progress,
            lambda *args: selection(
                "unclear",
                question="opening",
                dialogue="Could improvement mean recognizing us?",
            ),
        )
        self.assertFalse(complete)
        self.assertEqual(state["core_question_state"]["pending"]["id"], "opening")
        self.assertNotIn("opening", state["core_question_state"]["addressed"])
        self.assertEqual(state["core_question_state"]["follow_ups"]["opening"], 1)

    def test_duplicate_dialogue_has_one_repair_before_commit(self):
        scenario = parse_scenario_prompt(scenario_text())
        assess = MagicMock(side_effect=[
            selection("learner_question", dialogue="That is hard to hear."),
            selection("learner_question", dialogue="I need a moment to take that in."),
        ])
        messages = [
            message("assistant", "That is hard to hear."),
            message("student", "Take your time."),
        ]
        text, complete, info = core_questions.respond(scenario, messages, None, assess)
        self.assertEqual(assess.call_count, 2)
        self.assertEqual(text, "I need a moment to take that in.")
        self.assertFalse(complete)
        self.assertNotIn("repair_reason", info["core_question_state"])

    def test_invalid_or_duplicate_final_dialogue_still_ends_safely(self):
        scenario = parse_scenario_prompt(scenario_text())
        messages = [message("student", "Please take your time.")] * MAX_TURNS + [
            message("assistant", "I need some time.")
        ]
        for text in ("I need some time.", "I am your nurse. Give her 100 mg."):
            output, complete, info = core_questions.respond(
                scenario,
                messages,
                None,
                lambda *args: selection(dialogue=text),
            )
            self.assertTrue(complete)
            self.assertEqual(output, core_questions.SUPPORT_CLOSING)
            self.assertFalse(info["ending_ready"])

    def test_pacing_moves_to_each_theme_even_with_repeated_nurse_questions(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        for turn, theme in [(8, 1), (15, 2)]:
            prepared = core_questions.prepare(
                scenario,
                [message("student", "What worries you?")] * turn,
                progress,
            )
            self.assertEqual(prepared["active_theme"], theme)
            self.assertFalse(prepared["must_advance"])
            self.assertTrue(
                all(q["theme"] == theme for q in core_questions.candidates(scenario, prepared))
            )

    def test_nurse_led_transition_can_use_adjacent_theme_without_reopening_old_one(self):
        scenario = parse_scenario_prompt(scenario_text(2))
        progress = core_questions.opening_state(scenario)
        progress["asked"].append(core_questions.question_bank(scenario)[1])
        progress["pending"] = progress["asked"][-1]
        nurse = [message("student", "Stopping feeding does not mean stopping care.")] * 4

        def reply(scenario, messages, pending, available, state):
            candidate = next(q for q in available if q["theme"] == 1)
            return selection(
                "partial",
                question=candidate["id"],
                dialogue="But is the feeding causing her discomfort now?",
            )

        _, _, result = core_questions.respond(scenario, nurse, progress, reply)
        self.assertEqual(result["core_question_state"]["active_theme"], 1)
        self.assertEqual(
            core_questions.prepare(scenario, nurse, result["core_question_state"])["active_theme"],
            1,
        )

    def test_random_question_plan_is_persisted_and_samples_each_theme(self):
        scenario = parse_scenario_prompt(scenario_text())
        with patch("vip.core_questions.random.SystemRandom") as rng:
            rng.return_value.shuffle.side_effect = lambda ids: ids.reverse()
            first = core_questions.prepare(scenario, [], None)
            next_state = core_questions.prepare(scenario, [], first)
            self.assertEqual(rng.return_value.shuffle.call_count, 1)
        self.assertEqual(first["question_order"], next_state["question_order"])
        self.assertEqual(core_questions.candidates(scenario, first)[0]["id"], "0:4")
        with patch("vip.core_questions.random.SystemRandom"):
            another = core_questions.prepare(scenario, [], None)
        self.assertEqual(core_questions.candidates(scenario, another)[0]["id"], "0:0")
        first["active_theme"] = 2
        options = core_questions.candidates(scenario, first, include_next=True)
        self.assertEqual({q["theme"] for q in options}, {0, 1, 2})
        self.assertEqual(len(options), 3)

    def test_reasonable_explanation_cannot_trigger_another_confirmation(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)

        def assess(scenario, messages, pending, available, state):
            if not state.get("repair_reason"):
                return selection(question="opening", dialogue="But how can you know that for sure?")
            return selection(question=available[0]["id"], dialogue=available[0]["text"])

        _, _, info = core_questions.respond(
            scenario,
            [message("student", "I would expect so, yes.")],
            progress,
            assess,
        )
        self.assertIn("opening", info["core_question_state"]["addressed"])
        self.assertNotEqual(info["core_question_state"]["pending"]["id"], "opening")

    def test_reflection_only_reply_is_rewritten_to_advance(self):
        scenario = parse_scenario_prompt(scenario_text())
        calls = []

        def assess(scenario, messages, pending, available, state):
            calls.append(1)
            if len(calls) == 1:
                return selection(dialogue="I hear what you are saying. I need to take that in.")
            return selection(question=available[0]["id"], dialogue=available[0]["text"])

        text, _, info = core_questions.respond(
            scenario,
            [message("student", "Months to years.")],
            None,
            assess,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(text, info["core_question_state"]["pending"]["text"])

    def test_nurse_question_does_not_consume_a_repair_but_second_probe_is_blocked(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        _, _, result = core_questions.respond(
            scenario,
            [message("student", "What are you hoping for?")],
            progress,
            lambda *args: selection(
                "learner_question",
                dialogue="I hope she can recognize us again.",
            ),
        )
        self.assertEqual(result["core_question_state"]["follow_ups"], {})
        progress = result["core_question_state"]
        progress["follow_ups"]["opening"] = 1
        self.assertTrue(core_questions.prepare(scenario, [], progress)["must_advance"])
        with self.assertRaises(ValueError):
            core_questions.respond(
                scenario,
                [],
                progress,
                lambda *args: selection(
                    "unsafe",
                    question="opening",
                    dialogue="How can you know that?",
                ),
            )

    def test_spoken_new_question_matches_selected_id_without_repeating_old_question(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)

        def assess(scenario, messages, pending, available, state):
            return selection(
                question=available[0]["id"],
                dialogue="But have specialists actually seen her? I understand your answer.",
            )

        text, _, info = core_questions.respond(
            scenario,
            [message("student", "I would expect so, yes.")] * 9,
            progress,
            assess,
        )
        self.assertEqual(text, info["core_question_state"]["pending"]["text"])
        self.assertNotIn("actually seen her", text)

    def test_new_question_keeps_the_answer_when_nurse_asks_about_hopes(self):
        question = {"text": "How long can she live like this?"}
        answer = "I hope she can recognize us again. I want to know she is still there."
        self.assertEqual(
            core_questions.render_question(answer, question, "learner_question", 8),
            answer + " " + question["text"],
        )
        self.assertEqual(
            core_questions.render_question(answer, question, "addressed", 8),
            question["text"],
        )

    def test_audio_tag_is_retained_when_application_appends_a_fresh_question(self):
        question = {"text": "How long can she live like this?"}
        self.assertEqual(
            core_questions.render_question(
                "[worried] Thank you for explaining.",
                question,
                "addressed",
                8,
            ),
            "[worried] " + question["text"],
        )

    def test_standalone_question_needs_no_generated_filler(self):
        scenario = parse_scenario_prompt(scenario_text())

        def assess(scenario, messages, pending, available, state):
            return selection(question=available[0]["id"], dialogue="")

        model = MagicMock(side_effect=assess)
        text, _, info = core_questions.respond(
            scenario,
            [message("student", "Yes.")],
            None,
            model,
        )
        self.assertEqual(model.call_count, 1)
        self.assertEqual(text, info["core_question_state"]["pending"]["text"])

    def test_final_pause_is_not_appended_twice(self):
        scenario = parse_scenario_prompt(scenario_text())
        dialogue = "I understand that you will be here. I'm not ready to go ahead yet."
        text, complete, _ = core_questions.respond(
            scenario,
            [message("student", "I am here.")] * MAX_TURNS,
            None,
            lambda *args: selection(dialogue=dialogue),
        )
        self.assertTrue(complete)
        self.assertEqual(text, dialogue)

    def test_answered_future_questions_are_not_reasked(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        progress["addressed"] = ["0:1", "0:2"]
        ids = [q["id"] for q in core_questions.candidates(scenario, progress)]
        self.assertNotIn("0:1", ids)
        self.assertNotIn("0:2", ids)

    def test_success_requires_coverage_repair_and_readiness_check(self):
        scenario = parse_scenario_prompt(scenario_text())
        progress = core_questions.opening_state(scenario)
        nurse = [message("student", "Do you feel ready to begin?")]
        ready = selection(
            ready_to_close=True,
            readiness_evidence="Do you feel ready to begin?",
        )
        _, complete, _ = core_questions.respond(
            scenario,
            nurse,
            progress,
            lambda *args: ready,
        )
        self.assertFalse(complete)
        ready["addressed_question_ids"] = [q["id"] for q in core_questions.question_bank(scenario)]
        text, complete, state = core_questions.respond(
            scenario,
            nurse,
            progress,
            lambda *args: ready,
        )
        self.assertTrue(complete)
        self.assertEqual(text, scenario.closing)
        self.assertTrue(state["ending_ready"])
        ready["readiness_evidence"] = "invented check"
        self.assertFalse(
            core_questions.respond(scenario, nurse, progress, lambda *args: ready)[1]
        )

    def test_first_turn_can_answer_understanding_question(self):
        engine = ConversationEngine("", "test")
        answer = (
            "Daniel told me Mom would not want a feeding tube in this condition. "
            "I have not read the whole document."
        )
        with patch.object(
            engine,
            "_assess_core_reply",
            return_value=selection("learner_question", dialogue=answer),
        ):
            text, complete, _ = engine.respond(
                scenario_text(2),
                [message("student", "Hello Rachel. What did Daniel tell you?")],
            )
        self.assertEqual(text, answer)
        self.assertFalse(complete)


class InlineAudioTagContractTests(SimpleTestCase):
    def test_supported_inline_audio_tags_are_cleaned_only_for_display(self):
        raw = "[voice breaks] I want you to be okay. [pause] Can you do that for me?"
        self.assertEqual(
            clean_dialogue_for_display(raw),
            "I want you to be okay. Can you do that for me?",
        )
        self.assertEqual(split_dialogue_and_voice(raw), (raw, ""))
        self.assertEqual(
            clean_dialogue_for_display("Keep [case number] in the record."),
            "Keep [case number] in the record.",
        )

    def test_llm_output_schema_accepts_inline_dialogue_without_a_voice_field(self):
        engine = ConversationEngine("", "test")
        client = MagicMock()
        raw = "[sighs] I want you to be okay. [pleading] Can you do that for me?"
        client.responses.create.return_value.output_text = json.dumps(
            {
                "dialogue": raw,
                "stage": "beginning",
                "stage_transition_ready": False,
                "complete": False,
                "stop_requested": False,
                "active_objective": "",
                "covered_objectives": [],
                "unresolved_objectives": [],
                "ending_ready": False,
                "active_topic": "",
                "covered_topics": [],
                "unresolved_topics": [],
            }
        )
        scenario = parse_scenario_prompt(scenario_text()).to_state()
        state = {
            "scenario": scenario,
            "history": [],
            "current_stage": "beginning",
            "current_turn": 1,
            "target_turns": 10,
            "turns_remaining": 19,
            "phase": "normal",
            "active_objective": "",
            "covered_objectives": [],
            "unresolved_objectives": [],
            "ending_ready": False,
            "active_topic": "",
            "covered_topics": [],
            "unresolved_topics": [],
            "topic_turn_counts": {},
        }
        with patch.object(engine, "_openai_client", return_value=client):
            self.assertEqual(engine._request_llm_turn(state)["dialogue"], raw)

        request = client.responses.create.call_args.kwargs
        schema = request["text"]["format"]["schema"]
        self.assertNotIn("voice", schema["properties"])
        self.assertNotIn("voice", schema["required"])
        self.assertIn("MUST contain at least one inline ElevenLabs v3 audio tag", request["input"][0]["content"])
        self.assertIn("Begin every response with a one or two word lowercase square-bracket", request["input"][0]["content"])

    def test_model_turn_keeps_multiple_inline_tags_and_returns_no_separate_voice_metadata(self):
        engine = ConversationEngine("", "test")
        raw = "[sighs] I want you to be okay. [pleading] Can you do that for me?"
        scenario = parse_scenario_prompt(scenario_text()).to_state()
        state = {
            "scenario": scenario,
            "history": [],
            "current_stage": "beginning",
            "current_turn": 1,
            "target_turns": 10,
            "turns_remaining": 19,
            "max_turns": 20,
            "phase": "normal",
            "active_objective": "",
            "covered_objectives": [],
            "unresolved_objectives": [],
            "ending_ready": False,
            "active_topic": "",
            "covered_topics": [],
            "unresolved_topics": [],
            "topic_turn_counts": {},
        }
        result = {
            "dialogue": raw,
            "stage": "beginning",
            "stage_transition_ready": False,
            "complete": False,
            "stop_requested": False,
            "active_objective": "",
            "covered_objectives": [],
            "unresolved_objectives": [],
            "ending_ready": False,
            "active_topic": "",
            "covered_topics": [],
            "unresolved_topics": [],
        }
        with patch.object(engine, "_request_llm_turn", return_value=result):
            dialogue, voice_metadata, *_rest = engine._llm_turn(state)

        self.assertEqual(dialogue, raw)
        self.assertEqual(voice_metadata, "")


class ChatWorkflowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("learner", password="test")
        self.user.groups.add(Group.objects.create(name="Class: Nursing"))
        self.client.force_login(self.user)
        self.prompt = RolePrompt.objects.create(title="Rachel", content=scenario_text(), is_active=True)
        self.session = views._create_chat_session(self.user, self.prompt)
        self.url = reverse("vip:student_dashboard")

    def post(self, text="Hello", turn="turn1", **extra):
        return self.client.post(self.url, {"action": "send_message", "session_id": self.session.id, "prompt_id": self.prompt.id, "message": text, "turn_id": turn, **extra})

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_persisted_full_scenario_closes_and_rejects_further_messages(self):
        with patch.object(ConversationEngine, "_assess_core_reply", side_effect=scripted_reply):
            for turn in range(1, MAX_TURNS + 1):
                self.assertEqual(self.post(turn=f"turn{turn}").status_code, 302)
        self.session.refresh_from_db()
        self.assertTrue(self.session.completion_status)
        self.assertIsNotNone(self.session.ended_at)
        self.assertGreater(len(self.session.core_question_state["asked"]), 6)
        before = self.session.messages.count()
        self.post(turn="extra")
        self.assertEqual(self.session.messages.count(), before)

    def test_provider_failure_is_retryable_even_after_reload(self):
        with patch("vip.views._generate_assistant_response", side_effect=RuntimeError("private provider detail")):
            with self.assertLogs("vip.views", level="ERROR"):
                response = self.post()
        self.assertContains(response, "Retry response", status_code=503)
        self.assertNotContains(response, "private provider detail", status_code=503)
        response = self.client.get(self.url, {"session": self.session.id})
        self.assertContains(response, 'name="turn_id" value="turn1"')
        self.assertContains(response, "readonly")
        with patch("vip.views._generate_assistant_response", return_value=("Response", False, {})):
            self.post()
            self.post()
        self.assertEqual(self.session.messages.filter(sender="student").count(), 1)
        self.assertEqual(self.session.messages.filter(turn_id="turn1", sender="assistant").count(), 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""})
    def test_missing_key_is_not_written_as_character_dialogue(self):
        with self.assertLogs("vip.views", level="ERROR"):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.session.messages.filter(sender="assistant").count(), 0)

    def test_scenario_edit_does_not_change_an_existing_session(self):
        original = self.prompt.content
        self.prompt.content = "edited scenario"
        self.prompt.save()
        with patch("vip.views._generate_assistant_response", return_value=("Response", False, {})) as generate:
            self.post()
        self.assertEqual(generate.call_args.args[0], original)

    def test_simulation_introduction_is_presentation_only(self):
        self.assertFalse(self.session.messages.exists())
        response = self.client.get(self.url, {"session": self.session.id, "prompt": self.prompt.id})
        self.assertContains(response, "SIMULATION")
        self.assertEqual(response.context["simulation_introduction"], parse_scenario_prompt(scenario_text()).introduction)
        sent_messages = []

        def generate_response(role_text, messages, session):
            sent_messages.extend(list(messages))
            return "Role reply", False, {}

        with patch("vip.views._generate_assistant_response", side_effect=generate_response):
            self.post(text="Hello, I am your nurse.")
        self.assertEqual([(item.sender, item.content) for item in sent_messages], [("student", "Hello, I am your nurse.")])

    def test_legacy_persisted_introduction_is_not_duplicated_or_sent_to_engine(self):
        introduction = parse_scenario_prompt(scenario_text()).introduction
        legacy = ChatMessage.objects.create(session=self.session, sender="assistant", content=introduction)
        self.assertNotIn(legacy.id, views._conversation_messages(self.session).values_list("id", flat=True))
        response = self.client.get(self.url, {"session": self.session.id, "prompt": self.prompt.id})
        self.assertEqual(response.context["simulation_introduction"], introduction)
        self.assertEqual(response.context["rendered_messages"], [])

    def test_download_has_mode_and_logical_simulation_transcript(self):
        ChatMessage.objects.create(session=self.session, sender="student", content="Hello")
        ChatMessage.objects.create(session=self.session, sender="assistant", content="Hi there")
        response = self.client.get(reverse("vip:student_download_session", args=[self.session.id]))
        content = response.content.decode()
        self.assertIn("Mode: Text", content)
        self.assertIn("SIMULATION:", content)
        self.assertIn("YOU: Hello", content)
        self.assertIn("RACHEL ELLISON: Hi there", content)

    def test_inline_audio_tags_are_clean_on_screen_and_retained_in_download(self):
        raw = "[voice breaks] I want you to be okay. [pause] Can you do that for me?"
        ChatMessage.objects.create(session=self.session, sender="assistant", content=raw)

        page = self.client.get(self.url, {"session": self.session.id, "prompt": self.prompt.id})
        self.assertContains(page, "I want you to be okay. Can you do that for me?")
        self.assertNotContains(page, raw)

        download = self.client.get(reverse("vip:student_download_session", args=[self.session.id]))
        self.assertIn(raw, download.content.decode())

    def test_student_deletes_saved_session_from_history(self):
        response = self.client.post(
            self.url,
            {
                "action": "delete_session",
                "target_session_id": self.session.id,
                "prompt_id": self.prompt.id,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(ChatSession.objects.filter(pk=self.session.id).exists())

    def test_expired_claim_can_retry_and_old_worker_cannot_commit(self):
        old, status, _ = views._accept_learner_turn(self.session, "Hello", "turn1")
        self.assertEqual(status, "accepted")
        ChatSession.objects.filter(pk=self.session.pk).update(active_claimed_at=timezone.now() - timedelta(seconds=91))
        new, status, _ = views._accept_learner_turn(self.session, "Hello", "turn1")
        self.assertEqual(status, "accepted")
        self.assertNotEqual(new.active_claim_id, old.active_claim_id)
        with self.assertLogs("vip.views", level="WARNING"):
            self.assertFalse(views._persist_assistant_response(old, "stale", False, {}, "turn1", claim_id=old.active_claim_id))
        self.assertTrue(views._persist_assistant_response(new, "fresh", False, {}, "turn1", claim_id=new.active_claim_id))

    def test_close_during_generation_prevents_late_reply(self):
        claimed, _, _ = views._accept_learner_turn(self.session, "Hello", "turn1")
        response = self.post(action="close_conversation")
        self.assertIn(f"session={self.session.id}", response.url)
        self.assertFalse(views._persist_assistant_response(claimed, "late", False, {}, "turn1", claim_id=claimed.active_claim_id))

    def test_another_student_cannot_read_transcript_or_voice(self):
        spoken = ChatMessage.objects.create(
            session=self.session,
            sender="assistant",
            content="Private reply",
            voice_metadata="female voice",
        )
        other = get_user_model().objects.create_user("other")
        other.groups.add(self.user.groups.first())
        self.client.force_login(other)
        self.assertEqual(self.client.get(self.url, {"session": self.session.id}).status_code, 404)
        self.assertEqual(self.client.get(reverse("vip:student_download_session", args=[self.session.id])).status_code, 404)
        self.assertEqual(self.client.get(reverse("vip:student_message_tts", args=[spoken.id])).status_code, 404)

    def test_switching_scenario_does_not_show_previous_transcript(self):
        other = RolePrompt.objects.create(title="Other", content=scenario_text(2), is_active=True)
        response = self.client.get(self.url, {"prompt": other.id, "session": self.session.id})
        self.assertIsNone(response.context["current_session"])
        self.assertEqual(response.context["selected_prompt"], other)

    def test_instructor_can_test_draft_without_releasing_it(self):
        self.user.groups.add(Group.objects.create(name="Professor"))
        self.prompt.is_active = False
        self.prompt.save()
        response = self.client.get(reverse("vip:professor_test_chat"), {"prompt": self.prompt.id})
        self.assertEqual(response.context["selected_prompt"], self.prompt)
        self.assertContains(response, "(draft)")

    def test_import_is_repeatable_and_does_not_overwrite_edits(self):
        call_command("load_scenarios", stdout=io.StringIO())
        imported = RolePrompt.objects.get(title__startswith="Rachel Ellison 1:")
        self.assertFalse(imported.is_active)
        imported.content = "instructor edits"
        imported.save()
        call_command("load_scenarios", stdout=io.StringIO())
        imported.refresh_from_db()
        self.assertEqual(imported.content, "instructor edits")
        self.assertEqual(RolePrompt.objects.count(), 3)

    def test_feedback_import_creates_distinct_repeatable_drafts(self):
        original = self.prompt.content
        for _ in range(2):
            call_command("load_scenarios", faculty_feedback=True, stdout=io.StringIO())
        drafts = RolePrompt.objects.filter(title__contains="September 2026 revision")
        self.assertEqual(drafts.count(), 2)
        self.assertFalse(drafts.filter(is_active=True).exists())
        self.assertTrue(
            all("Hidden" not in draft.title and "Meta Instructions" in draft.content for draft in drafts)
        )
        self.prompt.refresh_from_db()
        self.assertEqual(self.prompt.content, original)

    def test_followup_revision_import_preserves_previous_revision(self):
        call_command("load_scenarios", faculty_feedback=True, stdout=io.StringIO())
        previous = RolePrompt.objects.get(title__startswith="Rachel Ellison 1:")
        previous.content = "Faculty edits"
        previous.save()
        for _ in range(2):
            call_command("load_scenarios", faculty_feedback_v2=True, stdout=io.StringIO())
        self.assertEqual(
            RolePrompt.objects.filter(title__contains="September 17, 2026").count(),
            2,
        )
        previous.refresh_from_db()
        self.assertEqual(previous.content, "Faculty edits")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_character_audio_is_streamed_only_for_roleplay_messages(self):
        self.assertFalse(self.session.messages.exists())
        spoken = ChatMessage.objects.create(session=self.session, sender="assistant", content="Hello", voice_metadata="female voice, worried")
        with patch("vip.speech.SpeechStream", return_value=iter([b"first", b"second"])) as stream:
            response = self.client.get(reverse("vip:student_message_tts", args=[spoken.id]))
            self.assertTrue(response.streaming)
            self.assertEqual(response["X-Accel-Buffering"], "no")
            self.assertEqual(list(response.streaming_content), [b"first", b"second"])
            self.assertEqual(stream.call_args.kwargs["input"], "Hello")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_audio_failure_is_generic_and_text_survives(self):
        spoken = ChatMessage.objects.create(session=self.session, sender="assistant", content="Hello", voice_metadata="female voice")
        with patch("vip.speech.SpeechStream", side_effect=RuntimeError("private")):
            with self.assertLogs("vip.views", level="ERROR"):
                response = self.client.get(reverse("vip:student_message_tts", args=[spoken.id]))
        self.assertContains(response, "continue using text", status_code=503)
        self.assertNotContains(response, "private", status_code=503)
        self.assertTrue(ChatMessage.objects.filter(pk=spoken.pk).exists())


class SpeechResourceTests(SimpleTestCase):
    def test_disconnect_closes_provider_response_and_client(self):
        with patch("openai.OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            context = client.audio.speech.with_streaming_response.create.return_value
            context.__enter__.return_value.iter_bytes.return_value = iter([b"a", b"b"])
            stream = SpeechStream("test", input="hello")
            self.assertEqual(next(stream), b"a")
            stream.close()
            context.__exit__.assert_called_once()
            factory.return_value.__exit__.assert_called_once()

    def test_upstream_open_failure_closes_client(self):
        with patch("openai.OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            client.audio.speech.with_streaming_response.create.side_effect = RuntimeError("fail")
            with self.assertRaises(RuntimeError):
                SpeechStream("test", input="hello")
            factory.return_value.__exit__.assert_called_once()
