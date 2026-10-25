import unittest
from types import SimpleNamespace

import torch

from src.safetriage_utils import detect_language, predict_triage, explain_text_perturbation


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}


class DummyModel:
    config = SimpleNamespace(id2label={i: str(i + 1) for i in range(5)})

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(
            logits=torch.tensor([[0.1, 0.2, 2.5, 0.3, 0.1]], dtype=torch.float32)
        )


class SafeTriageUtilityTests(unittest.TestCase):
    def test_detect_english(self):
        self.assertEqual(detect_language("The patient has fever and cough today."), "English")

    def test_detect_urdu_script(self):
        self.assertEqual(detect_language("مجھے بخار ہے"), "Urdu")

    def test_detect_roman_urdu(self):
        self.assertEqual(detect_language("mujhe bukhar hai aur dard hai"), "Roman Urdu")

    def test_predict_triage_returns_valid_level_and_probabilities(self):
        level, confidence, probabilities = predict_triage(
            "The patient has fever and cough.", DummyTokenizer(), DummyModel()
        )
        self.assertEqual(level, 3)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertAlmostEqual(float(probabilities.sum().item()), 1.0, places=5)

    def test_perturbation_explanation_returns_word_scores(self):
        rows = explain_text_perturbation(
            "fever and cough today", DummyTokenizer(), DummyModel(), top_n=3
        )
        self.assertTrue(rows)
        self.assertIn("Probability drop", rows[0])
        self.assertLessEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
