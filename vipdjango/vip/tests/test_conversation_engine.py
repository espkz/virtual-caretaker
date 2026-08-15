import unittest
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path

from vip.conversation_engine import (
    ConversationEngine,
    _history,
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

    def test_parser_preserves_main_and_introduction_voice_settings(self):
        self.assertEqual(self.scenario.voice_gender, "female")
        self.assertEqual(self.scenario.voice_style, "tense, worried, emotionally tired, direct but not hostile, natural pauses")
        self.assertEqual(self.scenario.introduction_voice_gender, "male")
        self.assertIn("friendly instructional tone", self.scenario.introduction_voice_style)

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
        self.assertNotIn("## Middle", prompt)

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
        self.assertEqual(stage, "beginning")
        self.assertFalse(complete)
        self.assertFalse(interrupted)
        self.assertFalse(debug["stage_transition_ready"])
        self.assertTrue(debug["embedded_voice_removed"])

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
    def test_hard_cap_is_only_a_late_safety_fallback(self, openai_client):
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
        self.assertEqual(dialogue, self.scenario.closing)
        self.assertEqual(stage, "ending")
        self.assertTrue(complete)
        self.assertEqual(debug["reason"], "hard_budget_fallback")

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


if __name__ == "__main__":
    unittest.main()
