import unittest
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vip.conversation_engine import (
    DEFAULT_INTRODUCTION,
    ConversationEngine,
    _context_measurement,
    _history,
    _initial_objective_progress,
    _normalize_objective_progress,
    _objective_progress_context,
    _conversation_memory,
    _question_similarity,
    _voice_metadata,
    format_history_content,
    format_voice_metadata,
    split_dialogue_and_voice,
)
from vip.conversation_scenario import DEFAULT_LEARNER_ROLE, parse_scenario_prompt


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Message:
    sender: str
    content: str
    voice_metadata: str = ""


class ConversationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.role_text = (ROOT / "archived" / "prompts" / "rachel_ellison_prompt.md").read_text(encoding="utf-8")
        cls.scenario = parse_scenario_prompt(cls.role_text)
        cls.objective_role_text = cls.role_text + """

## Conversation Objectives
### Understand the situation
Possible expressions:
- What happened?
- What does this mean for her?
Resolved when:
The learner has explained the situation honestly and acknowledged uncertainty.

### Prepare for next steps
Possible concerns:
- What should I do at home?
Resolved when:
The learner has described a reasonable next step or the character has decided to continue learning.
"""

    def test_parser_reads_generic_objective_blocks_and_round_trips_state(self):
        scenario = parse_scenario_prompt(self.objective_role_text)

        self.assertEqual([objective.id for objective in scenario.objectives], [
            "understand-the-situation",
            "prepare-for-next-steps",
        ])
        self.assertEqual(scenario.objectives[0].possible_expressions, "What happened?\nWhat does this mean for her?")
        self.assertIn("explained the situation honestly", scenario.objectives[0].resolved_when)
        state = scenario.to_state()
        self.assertEqual(state["objectives"][1]["id"], "prepare-for-next-steps")
        self.assertEqual(
            [objective.id for objective in scenario.from_state(state).objectives],
            ["understand-the-situation", "prepare-for-next-steps"],
        )

    def test_objective_progress_context_is_guidance_not_a_rigid_script(self):
        scenario = parse_scenario_prompt(self.objective_role_text)
        context = _objective_progress_context(scenario.to_state(), {})

        self.assertIn("understand-the-situation", context)
        self.assertIn("Possible expressions", context)
        self.assertIn("Resolved when", context)
        self.assertIn("not a rigid checklist", context)
        self.assertIn("One learner response may cover multiple objectives", context)
        self.assertIn("do not force every objective", context)

    def test_objectives_are_layered_with_character_personality_guidance(self):
        scenario = parse_scenario_prompt(self.objective_role_text)
        reserved_state = scenario.to_state()
        reserved_state["beginning"] = "The character is reserved, precise, and asks for time before responding."
        expressive_state = scenario.to_state()
        expressive_state["beginning"] = "The character is openly emotional, warm, and speaks in short bursts."
        engine = ConversationEngine("global prompt", "unused")

        reserved_prompt = engine._system_prompt(reserved_state, {"current_stage": "beginning"})
        expressive_prompt = engine._system_prompt(expressive_state, {"current_stage": "beginning"})

        self.assertIn("understand-the-situation", reserved_prompt)
        self.assertIn("understand-the-situation", expressive_prompt)
        self.assertIn(reserved_state["beginning"], reserved_prompt)
        self.assertIn(expressive_state["beginning"], expressive_prompt)
        self.assertNotEqual(reserved_prompt, expressive_prompt)

    def test_objective_progress_allows_multiple_and_out_of_order_coverage(self):
        scenario = parse_scenario_prompt(self.objective_role_text)
        state = _initial_objective_progress(scenario.to_state(), {})
        progress = _normalize_objective_progress(
            scenario.to_state(),
            state,
            {
                "active_objective": "understand-the-situation",
                "covered_objectives": ["prepare-for-next-steps", "not-a-real-objective"],
                "unresolved_objectives": ["understand-the-situation"],
                "ending_ready": False,
            },
            [],
        )

        self.assertEqual(progress["covered_objectives"], ["prepare-for-next-steps"])
        self.assertEqual(progress["unresolved_objectives"], ["understand-the-situation"])
        self.assertEqual(progress["active_objective"], "understand-the-situation")

    def test_objective_progress_keeps_prior_unresolved_concerns_until_covered(self):
        scenario = parse_scenario_prompt(self.objective_role_text)
        state = {
            "active_objective": "understand-the-situation",
            "covered_objectives": [],
            "unresolved_objectives": ["understand-the-situation", "prepare-for-next-steps"],
        }
        progress = _normalize_objective_progress(
            scenario.to_state(),
            state,
            {
                "active_objective": "understand-the-situation",
                "covered_objectives": [],
                "unresolved_objectives": [],
                "ending_ready": False,
            },
            [{"role": "user", "content": "LEARNER:\nI am still worried about what happened."}],
        )

        self.assertEqual(
            progress["unresolved_objectives"],
            ["understand-the-situation", "prepare-for-next-steps"],
        )
        self.assertEqual(progress["active_objective"], "understand-the-situation")
        self.assertEqual(progress["recent_topics"], ["I am still worried about what happened."])

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

    def test_missing_learner_role_uses_generic_fallback(self):
        role_text = self.role_text.replace(
            "## Learner Role\nHome-health registered nurse teaching Rachel how to recognize the need for tracheostomy suctioning, perform suctioning, and administer prescribed free-water flushes through the PEG tube\n",
            "## Learner Role\n\n",
        )

        scenario = parse_scenario_prompt(role_text)

        self.assertEqual(scenario.learner, DEFAULT_LEARNER_ROLE)
        self.assertNotIn("nursing student", scenario.learner.lower())

    def test_global_prompt_keeps_character_identity_and_human_perspective(self):
        global_prompt = (ROOT / "prompts" / "global_prompt.md").read_text(encoding="utf-8")
        self.assertIn("Play only the Simulated Character", global_prompt)
        self.assertIn("Never speak, think, act, or answer on behalf of the User", global_prompt)
        self.assertIn("Goals are conversational topics, not a rigid questionnaire", global_prompt)
        self.assertIn("Do not repeatedly ask about a topic", global_prompt)
        self.assertIn("End naturally", global_prompt)

    def test_new_role_prompts_parse_with_all_active_conversation_sections(self):
        for path in sorted((ROOT / "prompts").glob("role_rachel_ellison_*.md")):
            scenario = parse_scenario_prompt(path.read_text(encoding="utf-8"))
            self.assertTrue(scenario.character)
            self.assertTrue(scenario.learner)
            self.assertTrue(scenario.middle)
            self.assertTrue(scenario.ending)
            self.assertTrue(scenario.beginning_to_middle_cues)
            self.assertTrue(scenario.middle_to_ending_cues)
            self.assertEqual(len(scenario.objectives), 3)

    def test_system_prompt_uses_parsed_context_and_active_stage_not_raw_markdown(self):
        engine = ConversationEngine("global prompt", "unused")
        scenario = parse_scenario_prompt(
            (ROOT / "prompts" / "role_rachel_ellison_2.md").read_text(encoding="utf-8")
        )
        prompt = engine._system_prompt(
            scenario.to_state(),
            {
                "role_text": self.role_text,
                "current_stage": "middle",
            },
        )
        self.assertIn("SCENARIO REFERENCE DATA", prompt)
        self.assertIn(scenario.background_context, prompt)
        self.assertNotIn("SCENARIO ROLE PROMPT", prompt)
        self.assertNotIn(self.role_text, prompt)
        self.assertNotIn("## Conversation Goals", prompt)
        self.assertIn("CURRENT STAGE: middle", prompt)
        self.assertIn(scenario.middle, prompt)
        self.assertIn(scenario.middle_to_ending_cues, prompt)
        self.assertIn("The Learner drives the encounter", prompt)
        self.assertIn("never changes speaker ownership", prompt)
        self.assertIn("author notes are internal simulation data, not automatic character knowledge or dialogue", prompt)
        self.assertIn("Do not recite scenario facts or introduce clinical information unprompted", prompt)

    def test_objective_and_stage_guidance_are_character_side_not_facilitator_instructions(self):
        scenario = parse_scenario_prompt(self.objective_role_text)
        engine = ConversationEngine("global prompt", "unused")
        prompt = engine._system_prompt(
            scenario.to_state(),
            {"current_stage": "beginning", "recent_topics": []},
        )

        self.assertIn("ACTIVE-STAGE CHARACTER GUIDANCE", prompt)
        self.assertIn("not a learner questionnaire or facilitator task list", prompt)
        self.assertIn("Possible expressions are examples of concerns the Simulated Character may express", prompt)
        self.assertIn("They are not questions for the Learner's role", prompt)

    def test_graph_state_keeps_raw_role_markdown_out_of_runtime_state(self):
        engine = ConversationEngine("global prompt", "unused")
        graph_result = {
            "response": "I am still worried.",
            "completion_status": False,
            "current_stage": "beginning",
            "phase": "normal",
            "voice_metadata": "female voice, worried",
            "active_objective": "",
            "covered_objectives": [],
            "unresolved_objectives": [],
            "recent_topics": [],
            "ending_ready": False,
            "debug_info": {},
        }
        messages = [
            Message("student", "Hello."),
            Message("assistant", self.scenario.opening_line),
            Message("student", "I am worried about her."),
        ]

        with patch.object(engine.graph, "invoke", return_value=graph_result) as invoke:
            engine.respond(self.role_text, messages)

        graph_input = invoke.call_args.args[0]
        self.assertNotIn("role_text", graph_input)
        self.assertEqual(graph_input["scenario"]["character"], self.scenario.character)
        self.assertIn("background_context", graph_input["scenario"])

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
        self.assertIn("SIMULATED CHARACTER / ASSISTANT CHARACTER", prompt)
        self.assertIn("The human-controlled participant is the Learner", prompt)
        self.assertIn("PHASE CONTRACT", prompt)
        self.assertNotIn("## Middle", prompt)
        self.assertNotIn(self.scenario.middle, prompt)
        self.assertNotIn(self.scenario.ending, prompt)

    def test_context_measurement_counts_unicode_payload_without_retaining_content(self):
        measurement = _context_measurement(
            [
                {"role": "system", "content": "Café"},
                {"role": "user", "content": "Tell me more."},
            ],
            "generation",
        )

        self.assertEqual(measurement["request_kind"], "generation")
        self.assertEqual(measurement["message_count"], 2)
        self.assertEqual(measurement["content_chars"], len("CaféTell me more."))
        self.assertEqual(measurement["content_bytes"], len("CaféTell me more.".encode("utf-8")))
        self.assertGreater(measurement["serialized_bytes"], measurement["content_bytes"])
        self.assertGreater(measurement["estimated_tokens"], 0)
        self.assertNotIn("Café", measurement)

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
                {"role": "assistant", "content": "SIMULATED CHARACTER:\nIntroduction"},
                {"role": "user", "content": "LEARNER:\nHello"},
                {"role": "assistant", "content": "SIMULATED CHARACTER:\nI am worried."},
                {"role": "user", "content": "LEARNER:\nTell me more."},
            ],
        )
        legacy = [Message("student", "Hello"), Message("assistant", "I am worried.\n\n[female voice, tense]")]
        self.assertEqual(_history(legacy)[1]["content"], "SIMULATED CHARACTER:\nI am worried.")
        separated_but_leaking = [
            Message("assistant", "Introduction", "male voice, calm"),
            Message("student", "Hello"),
            Message("assistant", "I am worried. [tense, anxious]", "female voice, tense"),
        ]
        self.assertEqual(
            _history(separated_but_leaking)[-1]["content"],
            "SIMULATED CHARACTER:\nI am worried.",
        )

    def test_history_omits_application_owned_introduction(self):
        messages = [
            Message("assistant", self.scenario.introduction),
            Message("student", "Hello."),
            Message("assistant", self.scenario.opening_line),
        ]

        history = _history(messages, self.scenario)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "LEARNER:\nHello."},
                {"role": "assistant", "content": f"SIMULATED CHARACTER:\n{self.scenario.opening_line}"},
            ],
        )

    def test_history_labels_are_generic_and_keep_native_roles_across_scenarios(self):
        for filename in (
            "rachel_ellison_prompt.md",
            "caregiver_prompt.md",
            "peggy_collins_prompt.md",
        ):
            scenario = parse_scenario_prompt(
                (ROOT / "archived" / "prompts" / filename).read_text(encoding="utf-8")
            )
            history = _history(
                [
                    Message("student", "Hello."),
                    Message("assistant", scenario.opening_line),
                ],
                scenario,
            )

            self.assertEqual([item["role"] for item in history], ["user", "assistant"])
            self.assertTrue(history[0]["content"].startswith("LEARNER:\n"))
            self.assertTrue(history[1]["content"].startswith("SIMULATED CHARACTER:\n"))
            self.assertNotIn(scenario.character.splitlines()[0], history[1]["content"].splitlines()[0])

    def test_history_label_formatter_does_not_replace_native_roles(self):
        self.assertEqual(format_history_content("user", "Hello."), "LEARNER:\nHello.")
        self.assertEqual(
            format_history_content("assistant", "I am worried."),
            "SIMULATED CHARACTER:\nI am worried.",
        )

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
        _, _, _, _, _, debug = engine._llm_turn({
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
        self.assertEqual(payload[1]["content"], "SIMULATED CHARACTER:\nWelcome.")
        self.assertEqual(payload[2]["content"], "LEARNER:\nI am the nurse.")
        self.assertIn("HISTORY SPEAKER LABELS", payload[0]["content"])
        self.assertEqual(len(debug["llm_context"]), 1)
        self.assertEqual(debug["llm_context"][0]["request_kind"], "generation")
        self.assertEqual(debug["llm_context"][0]["message_count"], len(payload))
        self.assertEqual(
            debug["llm_context"][0]["content_chars"],
            sum(len(message["content"]) for message in payload),
        )

    @patch("openai.OpenAI")
    def test_llm_turn_returns_validated_objective_progress(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I hear that you need to understand what happened.","voice":"calm",'
            '"stage":"beginning","stage_transition_ready":false,"complete":false,"stop_requested":false,'
            '"active_objective":"understand-the-situation",'
            '"covered_objectives":["prepare-for-next-steps","unknown"],'
            '"unresolved_objectives":["understand-the-situation"],"ending_ready":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        _, _, _, complete, _, debug = engine._llm_turn({
            "scenario": parse_scenario_prompt(self.objective_role_text).to_state(),
            "history": [{"role": "user", "content": "I explained what has happened."}],
            "current_turn": 2,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 22,
            "current_stage": "beginning",
            "phase": "normal",
        })

        self.assertFalse(complete)
        self.assertEqual(debug["active_objective"], "understand-the-situation")
        self.assertEqual(debug["covered_objectives"], ["prepare-for-next-steps"])
        self.assertEqual(debug["unresolved_objectives"], ["understand-the-situation"])
        self.assertFalse(debug["ending_ready"])
        request_prompt = openai_client.return_value.responses.create.call_args.kwargs["input"][0]["content"]
        self.assertIn("OBJECTIVE PROGRESS", request_prompt)
        self.assertIn("understand-the-situation", request_prompt)
        self.assertIn("active_objective", request_prompt)

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
    def test_non_streaming_llm_turn_marks_first_output_and_completion(self, openai_client):
        response = openai_client.return_value.responses.create.return_value
        response.output_text = (
            '{"dialogue":"I hear you.","voice":"calm",'
            '"stage":"middle","stage_transition_ready":false,'
            '"complete":false,"stop_requested":false}'
        )
        response.usage = SimpleNamespace(input_tokens=321, output_tokens=18, total_tokens=339)
        timings = []
        engine = ConversationEngine("global prompt", "unused")
        _, _, _, _, _, debug = engine._llm_turn({
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
        self.assertFalse(openai_client.return_value.responses.create.call_args.kwargs.get("stream", False))
        self.assertEqual(debug["llm_context"][0]["input_tokens"], 321)
        self.assertEqual(debug["llm_context"][0]["output_tokens"], 18)
        self.assertEqual(debug["llm_context"][0]["total_tokens"], 339)

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
    def test_unresolved_objective_blocks_completion_without_ending_readiness(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I am still worried about that.","voice":"tense",'
            '"stage":"ending","stage_transition_ready":true,"complete":true,"stop_requested":false,'
            '"active_objective":"understand-the-situation","covered_objectives":[], '
            '"unresolved_objectives":["understand-the-situation","prepare-for-next-steps"],'
            '"ending_ready":false}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": parse_scenario_prompt(self.objective_role_text).to_state(),
            "history": [{"role": "user", "content": "Please keep explaining."}],
            "current_turn": 4,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 20,
            "current_stage": "middle",
            "phase": "normal",
        })

        self.assertEqual(dialogue, "I am still worried about that.")
        self.assertEqual(stage, "ending")
        self.assertFalse(complete)
        self.assertTrue(debug["completion_rejected"])
        self.assertEqual(debug["unresolved_objectives"], [
            "understand-the-situation",
            "prepare-for-next-steps",
        ])

    @patch("openai.OpenAI")
    def test_covered_objectives_allow_early_natural_completion(self, openai_client):
        openai_client.return_value.responses.create.return_value.output_text = (
            '{"dialogue":"I feel ready to continue learning.","voice":"calm",'
            '"stage":"ending","stage_transition_ready":true,"complete":true,"stop_requested":false,'
            '"active_objective":"","covered_objectives":["understand-the-situation","prepare-for-next-steps"],'
            '"unresolved_objectives":[],"ending_ready":true}'
        )
        engine = ConversationEngine("global prompt", "unused")
        dialogue, _, stage, complete, _, debug = engine._llm_turn({
            "scenario": parse_scenario_prompt(self.objective_role_text).to_state(),
            "history": [{"role": "user", "content": "I am ready."}],
            "current_turn": 4,
            "target_turns": 20,
            "max_turns": 24,
            "turns_remaining": 20,
            "current_stage": "middle",
            "phase": "normal",
        })

        self.assertEqual(dialogue, parse_scenario_prompt(self.objective_role_text).closing)
        self.assertEqual(stage, "ending")
        self.assertTrue(complete)
        self.assertTrue(debug["ending_ready"])

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

    def test_respond_passes_persisted_objective_progress_into_graph(self):
        engine = ConversationEngine("global prompt", "unused")
        messages = [
            Message("student", "first learner turn"),
            Message("assistant", "Opening line."),
            Message("student", "I understand the situation and want to discuss next steps."),
        ]
        with patch.object(engine.graph, "invoke") as invoke:
            invoke.return_value = {
                "response": "I hear you.",
                "voice_metadata": self.scenario.voice_metadata(),
                "current_stage": "beginning",
                "phase": "normal",
                "completion_status": False,
                "active_objective": "prepare-for-next-steps",
                "covered_objectives": ["understand-the-situation"],
                "unresolved_objectives": ["prepare-for-next-steps"],
                "recent_topics": ["I understand the situation."],
                "ending_ready": False,
                "debug_info": {},
            }
            response, complete, _ = engine.respond(
                self.objective_role_text,
                messages,
                conversation_state={
                    "current_stage": "beginning",
                    "active_objective": "prepare-for-next-steps",
                    "covered_objectives": ["understand-the-situation"],
                    "unresolved_objectives": ["prepare-for-next-steps"],
                    "recent_topics": ["What happened?"],
                    "ending_ready": False,
                },
            )

        graph_input = invoke.call_args.args[0]
        self.assertEqual(response, "I hear you.")
        self.assertFalse(complete)
        self.assertEqual(graph_input["active_objective"], "prepare-for-next-steps")
        self.assertEqual(graph_input["covered_objectives"], ["understand-the-situation"])
        self.assertEqual(graph_input["unresolved_objectives"], ["prepare-for-next-steps"])
        self.assertIn("understand-the-situation", graph_input["scenario"]["objectives"][0]["id"])

    def test_normal_turn_does_not_run_a_second_stop_classifier(self):
        engine = ConversationEngine("global prompt", "unused")
        messages = [Message("student", "first"), Message("student", "second")]

        with patch.object(engine, "_interrupt", side_effect=AssertionError("unexpected classifier")), patch.object(
            engine.graph, "invoke", return_value={
                "response": "I hear you.",
                "voice_metadata": "female voice, calm",
                "current_stage": "beginning",
                "phase": "normal",
                "completion_status": False,
                "context_measurements": [],
                "debug_info": {},
            },
        ) as invoke:
            response, complete, debug = engine.respond(self.role_text, messages)

        self.assertEqual(response, "I hear you.")
        self.assertFalse(complete)
        self.assertEqual(invoke.call_count, 1)

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
        self.assertEqual(
            [item["request_kind"] for item in debug["llm_context"]],
            ["generation", "repair"],
        )
        self.assertEqual(openai_client.call_count, 1)
        repair_prompt = openai_client.return_value.responses.create.call_args.kwargs["input"][0]["content"]
        self.assertIn("REPAIR INSTRUCTION", repair_prompt)
        self.assertIn("the simulated character's understanding", repair_prompt)
        self.assertNotIn("let Rachel's understanding", repair_prompt)

    def test_participant_ownership_is_scenario_driven_across_character_types(self):
        engine = ConversationEngine("global prompt", "unused")

        for filename in ("caregiver_prompt.md", "peggy_collins_prompt.md"):
            role_text = (ROOT / "archived" / "prompts" / filename).read_text(encoding="utf-8")
            scenario = parse_scenario_prompt(role_text)
            prompt = engine._system_prompt(
                scenario.to_state(),
                {"current_stage": "beginning"},
            )

            self.assertIn(scenario.character, prompt)
            self.assertIn(scenario.learner, prompt)
            self.assertIn("You are the Simulated Character", prompt)
            self.assertIn("the Learner", prompt)

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
