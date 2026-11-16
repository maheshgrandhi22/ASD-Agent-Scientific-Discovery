import json
import tempfile
import unittest
from pathlib import Path

from src.agent.research_session_memory import ResearchSessionMemory


class TestResearchSessionMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = ResearchSessionMemory(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_restore_session(self):
        created = self.memory.create_session("How can AI improve discovery?")
        restored = self.memory.get_session(created["session_id"])
        self.assertEqual(restored["research_question"],
                         "How can AI improve discovery?")
        self.assertEqual(restored["scientific_validation"], "not_performed")

    def test_session_survives_new_memory_instance(self):
        created = self.memory.create_session("Research question")
        restarted = ResearchSessionMemory(self.temp_dir.name)
        self.assertIsNotNone(restarted.get_session(created["session_id"]))

    def test_updates_persist_and_protect_identity(self):
        created = self.memory.create_session("Original question")
        updated = self.memory.update_session(created["session_id"], {
            "workflow_stage": "literature_review",
            "research_question": "Tampered question",
            "session_id": "tampered",
            "progress": {"papers_reviewed": 3},
        })
        self.assertEqual(updated["research_question"], "Original question")
        self.assertEqual(updated["session_id"], created["session_id"])
        self.assertEqual(updated["progress"]["papers_reviewed"], 3)

    def test_add_event(self):
        created = self.memory.create_session("Question")
        updated = self.memory.add_event(
            created["session_id"], "Literature review completed",
            {"verified_count": 2},
        )
        self.assertEqual(len(updated["events"]), 1)
        self.assertEqual(updated["events"][0]["details"]["verified_count"], 2)

    def test_list_sessions(self):
        self.memory.create_session("First question")
        self.memory.create_session("Second question")
        self.assertEqual(len(self.memory.list_sessions()), 2)

    def test_invalid_question_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.create_session("  ")

    def test_invalid_session_id_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.get_session("../unsafe")

    def test_missing_session_update_rejected(self):
        with self.assertRaises(KeyError):
            self.memory.update_session("0" * 32, {"status": "completed"})

    def test_invalid_update_rejected(self):
        created = self.memory.create_session("Question")
        with self.assertRaises(ValueError):
            self.memory.update_session(created["session_id"], [])

    def test_corrupt_session_file_rejected(self):
        created = self.memory.create_session("Question")
        path = Path(self.temp_dir.name) / f"{created['session_id']}.json"
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.memory.get_session(created["session_id"])


if __name__ == "__main__":
    unittest.main()
