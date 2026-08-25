import unittest
import threading
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vip.conversation_engine import (
    DEFAULT_INTRODUCTION,
    ConversationEngine,
    _history,
    _conversation_memory,
    _question_similarity,
    _voice_metadata,
    format_voice_metadata,
    split_dialogue_and_voice,
)
from vip.conversation_scenario import parse_scenario_prompt


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Message:
    sender: str
    content: str
    voice_metadata: str = ""


class ConversationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.role_text = (ROOT / "prompts" / "rachel_ellison_prompt.md").read_text(encoding="utf-8")
        cls.scenario = parse_scenario_prompt(cls.role_text)

    def test_parser_preserves_main_voice_settings_without_introduction_voice_fields(self):
        self.assertEqual(self.scenario.voice_gender, "female")
        self.assertEqual(self.scenario.voice_style, "tense, worried, emotionally tired, direct but not hostile, natural pauses")
        self.assertFalse(hasattr(self.scenario, "introduction_voice_gender"))
        self.assertFalse(hasattr(self.scenario, "introduction_voice_style"))

    def test_zero_turn_uses_requested_fallback_when_introduction_is_missing(self):
        role_text = self.role_text.replace(
            "## Introduction\nWelcome to the virtual patient simulation. Today, you are speaking with Rachel Ellison, the daughter of Margaret Ellison, a 66-year-old woman receiving home health care after a severe brain injury. Rachel is supposed to learn tracheostomy suctioning and PEG tube free-water flushes, but she has questions about her mother's condition before she can focus on the training. When you're ready, introduce yourself as the home-health nurse. Explain that you are here to help teach Rachel to participate in her mother’s care. Offer to answer any questions she has before you begin.\n",
            "## Introduction\n\n",
        )
        engine = ConversationEngine("global prompt", "unused")

        response, complete, debug = engine.respond(role_text, [])

        self.assertEqual(response, DEFAULT_INTRODUCTION)
        self.assertFalse(complete)
        self.assertEqual(debug["reason"], "introduction")

    def test_global_prompt_keeps_character_identity_and_human_perspective(self):
        global_prompt = (ROOT / "prompts" / "prompt_template.md").read_text(encoding="utf-8")
        self.assertIn("You are ALWAYS the simulated character", global_prompt)
        self.assertIn("You are NOT the learner, instructor, nurse", global_prompt)
        self.assertIn("Do not optimize", global_prompt)
        self.assertIn("Let emotion affect the wording", global_prompt)

    def test_voice_normalization_rejects_bare_gender(self):
        self.assertEqual(
            _voice_metadata("female", self.scenario),
            "female voice, tense, worried, emotionally tired, direct but not hostile, natural pauses",
        )
        self.assertEqual(
            _voice_metadata("frightened yet trying to maintain calm", self.scenario),
            "female voice, frightened yet trying to maintain calm",
        )
        self.assertEqual(
            format_voice_metadata(_voice_metadata("female", self.scenario)),
            "[female voice, tense, worried, emotionally tired, direct but not hostile, natural pauses]",
        )

    def test_context_contains_voice_style_and_stage_contract(self):
        engine = ConversationEngine("global prompt", "unused")
        prompt = engine._system_prompt(
            self.scenario.to_state(),
            {
                "current_stage": "beginning",
                "current_turn": 5,
                "target_turns": 20,
                "turns_remaining": 17,
                "phase": "normal",
            },
        )
        self.assertIn(self.scenario.voice_style, prompt)
        self.assertIn("Stage contract:", prompt)
        self.assertIn("stage_transition_ready", prompt)
        self.assertIn(self.scenario.beginning, prompt)
        self.assertIn("Never invent learner actions", prompt)
        self.assertIn("The target is an advisory pressure", prompt)
        self.assertIn("Never complete or emit the Closing merely because a number was reached", prompt)
        self.assertIn("scenario cues are semantic signals, not a checklist", prompt)
        self.assertIn("completion is semantic", prompt)
        self.assertIn("canonical Closing is an example/output template", prompt)
        self.assertIn("Questions are optional", prompt)
        self.assertNotIn("## Middle", prompt)

    def test_context_includes_answered_question_memory(self):
        engine = ConversationEngine("global prompt", "unused")
        prompt = engine._system_prompt(
            self.scenario.to_state(),
            {
                "current_stage": "middle",
                "current_turn": 8,
                "history": [
                    {"role": "assistant", "content": "What exact details should I share?"},
                    {"role": "user", "content": "Tell the doctor what has changed from baseline."},
                ],
            },
        )
        self.assertIn("The learner has already responded to these character questions", prompt)
        self.assertIn("What exact details should I share?", prompt)
        self.assertIn("move to a related concern or a new stage", prompt)

    def test_question_similarity_recognizes_rephrasing(self):
        self.assertGreaterEqual(
            _question_similarity(
                "What exact information should I give them about her breathing?",
                "What exact details should I share on the call about her breathing?",
            ),
            0.50,
        )

    def test_history_keeps_voice_separate_and_supports_legacy_embedded_voice(self):
        messages = [
            Message("assistant", "Introduction", "male voice, calm"),
            Message("student", "Hello"),
            Message("assistant", "I am worried.", "female voice, tense"),
            Message("student", "Tell me more."),
        ]
        self.assertEqual(
            _history(messages),
            [
                {"role": "assistant", "content": "Introduction"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "I am worried."},
                {"role": "user", "content": "Tell me more."},
            ],
        )
        legacy = [Message("student", "Hello"), Message("assistant", "I am worried.\n\n[female voice, tense]")]
        self.assertEqual(_history(legacy)[1]["content"], "I am worried.")
        separated_but_leaking = [
            Message("assistant", "Introduction", "male voice, calm"),
            Message("student", "Hello"),
            Message("assistant", "I am worried. [tense, anxious]", "female voice, tense"),
        ]
        self.assertEqual(_history(separated_but_leaking)[-1]["content"], "I am worried.")

    def test_legacy_voice_split_is_dialogue_boundary_only(self):
        self.assertEqual(
            split_dialogue_and_voice("I am scared.\n\n[female voice, tense]"),
            ("I am scared.", "female voice, tense"),
        )
        self.assertEqual(split_dialogue_and_voice("I am scared."), ("I am scared.", ""))

    @patch("openai.OpenAI")
    def test_llm_payload_keeps_identity_and_complete_native_history(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I am listening.","voice":"worried",'
            '"stage":"beginning","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [
                {"role": "assistant", "content": "Welcome."},
                {"role": "user", "content": "I am the nurse."},
            ],
            "current_turn": 2,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 22,
            "current_stage": "beginning",
            "phase": "normal",
        })
        payload = openai_client.return_value.responses.create.call_args.kwargs["input"]
        self.assertEqual([message["role"] for message in payload], ["system", "assistant", "user"])
        self.assertIn("ASSISTANT CHARACTER (## Role)", payload[0]["content"])
        self.assertIn("LEARNER / USER (## Learner Role)", payload[0]["content"])
        self.assertEqual(payload[1]["content"], "Welcome.")
        self.assertEqual(payload[2]["content"], "I am the nurse.")

    @patch("openai.OpenAI")
    def test_llm_response_keeps_dialogue_and_voice_separate(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I am scared.\\n\\n[female voice, tense]","voice":"female",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, voice, stage, complete, interrupted, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "Please tell me more."}],
            "current_turn": 5,
            "target_turns": 20,
            "max_turns": 22,
            "turns_remaining": 17,
            "current_stage": "beginning",
            "phase": "normal",
        })
        self.assertEqual(dialogue, "I am scared.")
        self.assertEqual(voice, "female voice, tense")
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)
        self.assertFalse(interrupted)
        self.assertTrue(debug["stage_transition_ready"])
        self.assertTrue(debug["embedded_voice_removed"])

    @patch("openai.OpenAI")
    def test_streaming_llm_turn_emits_dialogue_deltas_and_timing(self, openai_client):
        output = (
            '{"dialogue":"I hear you. I am listening.","voice":"calm",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        openai_client.return_value.responses.create.return_value = iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta=output[:24]),
                SimpleNamespace(type="response.output_text.delta", delta=output[24:]),
                SimpleNamespace(type="response.completed", delta=None),
            ]
        )
        deltas = []
        timings = []
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, _ = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "Please tell me more."}],
            "current_turn": 5,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 19,
            "current_stage": "beginning",
            "phase": "normal",
            "stream_callback": deltas.append,
            "timing_callback": timings.append,
        })
        self.assertEqual(dialogue, "I hear you. I am listening.")
        self.assertEqual("".join(item["text"] for item in deltas if item["kind"] == "gpt_delta"), dialogue)
        self.assertIn("gpt_request_start", timings)
        self.assertIn("gpt_first_output", timings)
        self.assertIn("gpt_completion", timings)
        self.assertTrue(openai_client.return_value.responses.create.call_args.kwargs["stream"])
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)

    @patch("openai.OpenAI")
    def test_non_streaming_llm_turn_marks_first_output_and_completion(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I hear you.","voice":"calm",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        timings = []
        engine = ConversationEngine("global prompt", "unused")
        engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "Please tell me more."}],
            "current_turn": 5,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 19,
            "current_stage": "beginning",
            "phase": "normal",
            "timing_callback": timings.append,
        })
        self.assertIn("gpt_request_start", timings)
        self.assertIn("gpt_first_output", timings)
        self.assertIn("gpt_completion", timings)
        self.assertLess(timings.index("gpt_first_output"), timings.index("gpt_completion"))

    @patch("openai.OpenAI")
    def test_completion_requires_semantic_ending_transition(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I am still worried.","voice":"tense",'
            '"stage":"beginning","stage_transition_ready":false,'
            '"complete":true,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "Please tell me more."}],
            "current_turn": 20,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 4,
            "current_stage": "beginning",
            "phase": "closure_preference",
        })
        self.assertEqual(dialogue, "I am still worried.")
        self.assertEqual(stage, "beginning")
        self.assertFalse(complete)
        self.assertTrue(debug["completion_rejected"])

    @patch("openai.OpenAI")
    def test_safety_cap_does_not_emit_fixed_closing(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I still have one more concern.","voice":"tense",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "I have one more question."}],
            "current_turn": 24,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 0,
            "current_stage": "middle",
            "phase": "closure_flexibility",
        })
        self.assertEqual(dialogue, "I still have one more concern.")
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)
        self.assertEqual(debug["reason"], "llm_turn_at_safety_cap")

    def test_missing_persisted_stage_does_not_use_turn_number(self):
        engine = ConversationEngine("global prompt", "unused")
        messages = [Message("student", f"learner turn {index}") for index in range(1, 16)]
        with patch.object(engine, "_interrupt", return_value=False), patch.object(engine.graph, "invoke") as invoke:
            invoke.return_value = {
                "response": "I am still Rachel.",
                "voice_metadata": self.scenario.voice_metadata(),
                "current_stage": "beginning",
                "phase": "closure_preference",
                "completion_status": False,
                "debug_info": {},
            }
            engine.respond(self.role_text, messages)
        self.assertEqual(invoke.call_args.args[0]["current_stage"], "beginning")

    def test_stream_callback_reaches_caller_before_graph_completion(self):
        engine = ConversationEngine("global prompt", "unused")
        messages = [
            Message("student", "first learner turn"),
            Message("student", "second learner turn"),
        ]
        callback_seen = threading.Event()
        release_graph = threading.Event()
        streamed = []

        def invoke(state):
            state["stream_callback"]({"kind": "gpt_delta", "text": "Hello."})
            callback_seen.set()
            release_graph.wait(timeout=1)
            return {
                "response": "Hello.",
                "voice_metadata": self.scenario.voice_metadata(),
                "current_stage": "beginning",
                "phase": "normal",
                "completion_status": False,
                "debug_info": {},
            }

        with patch.object(engine, "_interrupt", return_value=False), patch.object(
            engine.graph, "invoke", side_effect=invoke
        ):
            result = []
            worker = threading.Thread(
                target=lambda: result.append(
                    engine.respond(self.role_text, messages, stream_callback=streamed.append)
                )
            )
            worker.start()
            self.assertTrue(callback_seen.wait(timeout=1))
            self.assertTrue(worker.is_alive())
            release_graph.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(streamed, [{"kind": "gpt_delta", "text": "Hello."}])
        self.assertEqual(result[0][0], "Hello.")

    @patch("openai.OpenAI")
    def test_beginning_can_transition_to_middle_after_a_few_turns(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I understand. I am ready to talk about what comes next.","voice":"worried",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [
                {"role": "assistant", "content": "I am frightened. Is she going to wake up?"},
                {"role": "user", "content": "I understand why this is frightening. Recovery is unlikely, but we cannot claim certainty."},
            ],
            "current_turn": 3,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 21,
            "current_stage": "beginning",
            "phase": "normal",
        })
        self.assertIn("ready to talk", dialogue)
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)
        self.assertTrue(debug["stage_transition_ready"])

    @patch("openai.OpenAI")
    def test_semantic_ending_can_complete_before_soft_target(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I am ready.","voice":"calm",'
            '"stage":"ending","stage_transition_ready":true,'
            '"complete":true,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, _ = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "I am ready to learn."}],
            "current_turn": 3,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 21,
            "current_stage": "middle",
            "phase": "normal",
        })
        self.assertEqual(dialogue, self.scenario.closing)
        self.assertEqual(stage, "ending")
        self.assertTrue(complete)

    @patch("openai.OpenAI")
    def test_noncanonical_endpoint_is_semantically_verified(self, openai_client):
        openai_client.return_value.responses.create.side_effect = [
            type("Response", (), {"output_text": (
                '{"dialogue":"I appreciate you coming by. I\'m ready to start learning.","voice":"tense",'
                '"stage":"middle","stage_transition_ready":false,'
                '"complete":false,"stop_requested":false}'
            )})(),
            type("Response", (), {"output_text": (
                '{"ending_satisfied":true,"confidence":"high",'
                '"reason":"Rachel clearly indicates readiness for the next activity."}'
            )})(),
        ]
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [
                {"role": "assistant", "content": "Would you like to begin learning?"},
                {"role": "user", "content": "Yes, I think we can get started."},
            ],
            "current_turn": 18,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 6,
            "current_stage": "middle",
            "phase": "closure_preference",
        })
        self.assertEqual(dialogue, self.scenario.closing)
        self.assertEqual(stage, "ending")
        self.assertTrue(complete)
        self.assertEqual(debug["reason"], "semantic_ending_verified")
        self.assertTrue(debug["ending_check"]["satisfied"])
        self.assertEqual(openai_client.return_value.responses.create.call_count, 2)
        verifier_payload = openai_client.return_value.responses.create.call_args_list[1].kwargs["input"][1]["content"]
        self.assertIn("canonical_closing_example", verifier_payload)

    @patch("openai.OpenAI")
    def test_polite_intermediate_response_does_not_end(self, openai_client):
        openai_client.return_value.responses.create.side_effect = [
            type("Response", (), {"output_text": (
                '{"dialogue":"Thanks for explaining that. I still have a question about what happens next.",'
                '"voice":"worried","stage":"middle","stage_transition_ready":false,'
                '"complete":false,"stop_requested":false}'
            )})(),
            type("Response", (), {"output_text": (
                '{"ending_satisfied":false,"confidence":"high",'
                '"reason":"The character explicitly has an unresolved question."}'
            )})(),
        ]
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "I explained the current plan."}],
            "current_turn": 8,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 16,
            "current_stage": "middle",
            "phase": "normal",
        })
        self.assertIn("still have a question", dialogue)
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)
        self.assertFalse(debug["ending_check"]["satisfied"])
        self.assertEqual(openai_client.return_value.responses.create.call_count, 2)

    @patch("openai.OpenAI")
    def test_repeated_answered_question_gets_a_repair_turn(self, openai_client):
        openai_client.return_value.responses.create.side_effect = [
            type("Response", (), {"output_text": (
                '{"dialogue":"Okay, I understand. What exact details should I give them on the call to convey urgency?",'
                '"voice":"worried","stage":"middle","stage_transition_ready":false,'
                '"complete":false,"stop_requested":false}'
            )})(),
            type("Response", (), {"output_text": (
                '{"dialogue":"Okay, I understand. I will tell the doctor what has changed from her baseline. I am also worried about how long she may live like this.",'
                '"voice":"worried","stage":"middle","stage_transition_ready":false,'
                '"complete":false,"stop_requested":false}'
            )})(),
        ]
        history = [
            {"role": "assistant", "content": "What exact information should I give the on-call doctor?"},
            {"role": "user", "content": "Tell the doctor what has changed from her baseline."},
        ]
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, _, complete, _, debug = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": history,
            "current_turn": 3,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 21,
            "current_stage": "middle",
            "phase": "normal",
        })
        self.assertNotIn("What exact details should I give them on the call", dialogue)
        self.assertIn("what has changed from her baseline", dialogue)
        self.assertFalse(complete)
        self.assertTrue(debug["repetition_repaired"])
        self.assertEqual(openai_client.return_value.responses.create.call_count, 2)
        repair_prompt = openai_client.return_value.responses.create.call_args.kwargs["input"][0]["content"]
        self.assertIn("REPAIR INSTRUCTION", repair_prompt)

    @patch("openai.OpenAI")
    def test_no_filler_question_is_added_at_soft_target(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I understand. I need a moment to take that in.","voice":"tense",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, _ = engine._llm_turn({
            "scenario": self.scenario.to_state(),
            "history": [{"role": "user", "content": "I have answered your concern."}],
            "current_turn": 20,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 4,
            "current_stage": "middle",
            "phase": "closure_preference",
        })
        self.assertEqual(dialogue, "I understand. I need a moment to take that in.")
        self.assertEqual(stage, "middle")
        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
