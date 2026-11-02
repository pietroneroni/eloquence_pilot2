"""Regression tests for background evidence supplied in the opening message."""
import json
import unittest

from university_counseling import AgentWithRag as rag


class BackgroundCapturePromptTest(unittest.TestCase):
    def tearDown(self):
        rag.clear_last_rag_state()
        rag.clear_listener_patches()

    def test_opening_education_is_committed_when_background_advances(self):
        flow = rag.UniversityCounselorFlowOrchestrator(
            retriever=None,
            dialog_language="English",
        )
        flow._stage = "background"
        flow._opening_text = "I attended a language-focused high school."

        prompt = flow._background_prompt("I am interested in languages and communication.")
        self.assertIn("STUDENT_OPENING is student evidence too", prompt)
        self.assertIn("never ask for it again", prompt)

        control = {
            "intent": "background_valid",
            "patch": {"explicit": {"academic_background": "language-focused high school"}},
            "retrieval": {
                "query": "",
                "location": {"city": None, "region": None, "strict": False, "requirement": ""},
            },
        }
        rag.strip_and_apply_turn_control("<TURN_CONTROL>" + json.dumps(control) + "</TURN_CONTROL>")
        flow.instruct([], "I would like to learn more about languages.")

        self.assertEqual(
            flow._listener_memory["explicit"]["academic_background"],
            "language-focused high school",
        )


if __name__ == "__main__":
    unittest.main()
