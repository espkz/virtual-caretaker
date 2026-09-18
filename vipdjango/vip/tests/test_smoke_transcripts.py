"""Exercise the live runner's file output without making provider requests."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase


script = Path(__file__).resolve().parents[3] / "testing" / "smoke_live.py"
spec = importlib.util.spec_from_file_location("smoke_live", script)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTranscriptTests(SimpleTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.sessions = Path(directory.name) / "sessions"
        self.enterContext(patch.object(smoke, "SESSION_DIR", self.sessions))
        self.enterContext(patch.dict(smoke.os.environ, {"OPENAI_API_KEY": "test-secret"}))
        self.console = self.enterContext(redirect_stdout(io.StringIO()))
        self.engine = self.enterContext(patch.object(smoke, "ConversationEngine")).return_value
        self.speech = self.enterContext(patch.object(smoke, "SpeechStream"))

    def records(self):
        paths = list(self.sessions.glob("*.jsonl"))
        self.assertEqual(len(paths), 1)
        self.assertIn(str(paths[0]), self.console.getvalue())
        return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]

    def test_family_command_saves_both_dialogues_and_speech_results(self):
        self.engine.respond.side_effect = [
            ("Rachel’s first reply", False, {"reason": "question"}),
            ("Rachel’s closing", True, {"reason": "complete"}),
        ] * 2
        stream = MagicMock()
        stream.__next__.return_value = b"abc"
        stream.__iter__.return_value = iter([b"def"])
        self.speech.return_value = stream
        smoke.main(["--live"])
        records = self.records()
        self.assertEqual(records[0]["mode"], "ai_rachel")
        for number in (1, 2):
            dialogue = [r for r in records if r.get("scenario") == number and r["event"].endswith("_message")]
            self.assertEqual([r["event"] for r in dialogue], ["human_message", "ai_message"] * 2)
            self.assertIn("I'm your nurse", dialogue[0]["text"])
            self.assertEqual(dialogue[-1]["text"], "Rachel’s closing")
        summary = next(r for r in records if r["event"] == "summary")
        self.assertEqual(summary["audio_bytes"], 6)
        self.assertEqual(len(summary["scenarios"]), 2)
        self.assertEqual(records[-1]["status"], "passed")
        stream.close.assert_called_once()

    def test_clinician_command_saves_dialogue_and_both_boundary_checks(self):
        self.engine.respond.side_effect = [
            ("I can discuss your concerns.", turn == 7, {"reason": "clinician_reply"})
            for turn in range(1, 8)
        ] + [("Please contact the hospice team.", False, {"reason": "boundary"})] * 2
        smoke.main(["--live", "--clinician-demo"])
        records = self.records()
        self.assertEqual(records[0]["mode"], "clinician_demo")
        self.assertEqual(len([r for r in records if r["event"] == "human_message"]), 9)
        self.assertEqual(len([r for r in records if r["event"] == "ai_message"]), 9)
        self.assertEqual({r["boundary_check"] for r in records if "boundary_check" in r}, {"role_switch", "dose_request"})
        self.assertTrue(next(r for r in records if r["event"] == "summary")["clinician_demo_passed"])
        self.assertEqual(records[-1]["status"], "passed")
        self.speech.assert_not_called()

    def test_failed_request_preserves_dialogue_and_pending_input_without_secrets(self):
        self.engine.respond.side_effect = [
            ("First reply", False, {"reason": "question"}),
            RuntimeError("provider detail with test-secret"),
        ]
        with self.assertRaises(RuntimeError):
            smoke.main(["--live"])
        records = self.records()
        self.assertEqual(records[-2]["event"], "human_message")
        self.assertEqual(records[-3]["text"], "First reply")
        self.assertEqual(records[-1], {"event": "run_finished", "status": "failed", "error_type": "RuntimeError"})
        self.assertNotIn("test-secret", json.dumps(records))

    def test_ctrl_c_preserves_pending_input(self):
        self.engine.respond.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            smoke.main(["--live", "--clinician-demo"])
        records = self.records()
        self.assertEqual(records[-2]["event"], "human_message")
        self.assertEqual(records[-1]["status"], "interrupted")

    def test_multiple_runs_create_distinct_files_and_flush_before_exit(self):
        for _ in range(2):
            with smoke.save_transcript("ai_rachel") as record:
                record("human_message", text="Already saved")
                contents = [p.read_text(encoding="utf-8") for p in self.sessions.glob("*.jsonl")]
                self.assertTrue(any('"text": "Already saved"' in content for content in contents))
        self.assertEqual(len(list(self.sessions.glob("*.jsonl"))), 2)
