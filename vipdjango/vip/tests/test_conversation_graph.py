import unittest

from vip.conversation_graph import MAX_TURNS, TARGET_TURNS, _phase_for_turn, build_conversation_graph


class ConversationGraphTests(unittest.TestCase):
    def test_phase_is_pressure_not_stage_progression(self):
        self.assertEqual(_phase_for_turn(10), "normal")
        self.assertEqual(_phase_for_turn(17), "resolution_guidance")
        self.assertEqual(_phase_for_turn(TARGET_TURNS), "closure_preference")
        self.assertGreater(MAX_TURNS, TARGET_TURNS)

    def test_forward_stage_selection_is_the_semantic_transition_signal(self):
        def respond(state):
            return "response", "female voice, tense", "middle", False, False, {
                "stage_transition_ready": False,
                "reason": "llm_turn",
            }

        graph = build_conversation_graph(respond)
        result = graph.invoke({"current_turn": 5, "current_stage": "beginning", "max_turns": 22})
        self.assertEqual(result["current_stage"], "middle")
        self.assertTrue(result["stage_transition_ready"])
        self.assertEqual(result["voice_metadata"], "female voice, tense")

    def test_stage_does_not_regress_at_any_turn(self):
        def respond(state):
            return "response", "female voice, tense", "beginning", False, False, {
                "stage_transition_ready": True,
                "reason": "llm_turn",
            }

        graph = build_conversation_graph(respond)
        result = graph.invoke({"current_turn": 22, "current_stage": "middle", "max_turns": MAX_TURNS})
        self.assertEqual(result["current_stage"], "middle")
        self.assertFalse(result["stage_transition_ready"])

    def test_stage_can_advance_early_when_cues_are_ready(self):
        def respond(state):
            return "response", "female voice, tense", "middle", False, False, {
                "stage_transition_ready": True,
                "reason": "llm_turn",
            }

        graph = build_conversation_graph(respond)
        result = graph.invoke({"current_turn": 2, "current_stage": "beginning", "max_turns": MAX_TURNS})
        self.assertEqual(result["current_stage"], "middle")
        self.assertTrue(result["stage_transition_ready"])

    def test_target_turn_does_not_force_completion(self):
        def respond(state):
            return "continue", "female voice, tense", "middle", False, False, {
                "stage_transition_ready": True,
                "reason": "llm_turn",
            }

        graph = build_conversation_graph(respond)
        result = graph.invoke({
            "current_turn": TARGET_TURNS,
            "current_stage": "beginning",
            "max_turns": MAX_TURNS,
        })
        self.assertFalse(result["completion_status"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["current_stage"], "middle")

    def test_objective_progress_is_initialized_and_carried_through_graph(self):
        def respond(state):
            return "response", "female voice, calm", "beginning", False, False, {
                "active_objective": "next-steps",
                "covered_objectives": ["situation"],
                "unresolved_objectives": ["next-steps"],
                "recent_topics": ["what happens next"],
                "ending_ready": False,
            }

        graph = build_conversation_graph(respond)
        result = graph.invoke({
            "scenario": {
                "objectives": [
                    {"id": "situation"},
                    {"id": "next-steps"},
                ],
            },
            "current_turn": 2,
            "current_stage": "beginning",
            "max_turns": MAX_TURNS,
        })

        self.assertEqual(result["active_objective"], "next-steps")
        self.assertEqual(result["covered_objectives"], ["situation"])
        self.assertEqual(result["unresolved_objectives"], ["next-steps"])
        self.assertEqual(result["recent_topics"], ["what happens next"])
        self.assertFalse(result["ending_ready"])

    def test_objective_state_ignores_unknown_ids_and_selects_first_unresolved(self):
        def respond(state):
            return "response", "", "beginning", False, False, {}

        graph = build_conversation_graph(respond)
        result = graph.invoke({
            "scenario": {
                "objectives": [
                    {"id": "first"},
                    {"id": "second"},
                ],
            },
            "active_objective": "unknown",
            "covered_objectives": ["unknown"],
            "unresolved_objectives": ["second", "unknown"],
            "current_turn": 2,
            "current_stage": "beginning",
            "max_turns": MAX_TURNS,
        })

        self.assertEqual(result["active_objective"], "second")
        self.assertEqual(result["covered_objectives"], [])
        self.assertEqual(result["unresolved_objectives"], ["second"])


if __name__ == "__main__":
    unittest.main()
