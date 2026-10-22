"""Check the sampled gradient and refusal of incorrectly aligned scores."""
import unittest
import torch
from common import choose_training_ids, extract_teacher_scores, importance_loss


class ObjectiveTests(unittest.TestCase):
    def test_expected_sample_gradient_equals_exact_reverse_kl(self):
        # Enumerating every possible sample gives the exact expectation.
        logits = torch.tensor([0.3, -0.2, 1.1], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)
        logq = torch.tensor([0.55, 0.35, 0.10], dtype=torch.float64).log()
        old = logp.detach()
        losses, _ = importance_loss(logp, old, logq)
        sampled_expected = (old.exp() * losses).sum()
        sample_gradient = torch.autograd.grad(sampled_expected, logits, retain_graph=True)[0]
        exact = (logp.exp() * (logp - logq)).sum()
        exact_gradient = torch.autograd.grad(exact, logits)[0]
        torch.testing.assert_close(sample_gradient, exact_gradient, rtol=1e-12, atol=1e-12)

    def test_higher_teacher_probability_increases_sample_probability(self):
        logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)[0:1]
        loss, _ = importance_loss(logp, logp.detach(), torch.tensor([0.9], dtype=torch.float64).log())
        grad = torch.autograd.grad(loss.sum(), logits)[0]
        after = (logits.detach() - 0.1 * grad).softmax(-1)[0]
        self.assertGreater(after.item(), 0.5)

    def test_lower_teacher_probability_decreases_sample_probability(self):
        logits = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)[0:1]
        loss, _ = importance_loss(logp, logp.detach(), torch.tensor([0.1], dtype=torch.float64).log())
        grad = torch.autograd.grad(loss.sum(), logits)[0]
        after = (logits.detach() - 0.1 * grad).softmax(-1)[0]
        self.assertLess(after.item(), 0.5)

    def test_teacher_alignment_includes_first_response_token_and_eos(self):
        response = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:12", "token_id:99"],
                                                 "token_logprobs": [None, -0.2, -0.4]}}]}
        self.assertEqual(extract_teacher_scores(response, [11], [12, 99]), [-0.2, -0.4])
        response["choices"][0]["logprobs"]["tokens"][-1] = "token_id:100"
        with self.assertRaises(ValueError):
            extract_teacher_scores(response, [11], [12, 99])

    def test_nonfinite_teacher_score_is_rejected(self):
        response = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:12"],
                                                 "token_logprobs": [None, float("nan")]}}]}
        with self.assertRaises(ValueError):
            extract_teacher_scores(response, [11], [12])

    def test_training_selection_is_reproducible_and_excludes_evaluation(self):
        first = choose_training_ids(range(100), [100], [101], [5, 6], 42, 20)
        self.assertEqual(first, choose_training_ids(range(100), [100], [101], [5, 6], 42, 20))
        self.assertFalse(set(first) & {5, 6, 100, 101})
        with self.assertRaises(ValueError):
            choose_training_ids(range(100), [5], [101], [], 42, 20)

    def test_budget_extension_preserves_entire_pilot_prefix(self):
        prefix = choose_training_ids(range(800), [800], [801], range(50), 20261014, 64)
        larger = choose_training_ids(range(800), [800], [801], range(50), 20261014, 256, prefix)
        self.assertEqual(larger[:64], prefix)
        self.assertEqual(len(larger), len(set(larger)))
        self.assertFalse(set(larger) & set(range(50)))
        self.assertEqual(larger, choose_training_ids(range(800), [800], [801], range(50), 20261014, 256, prefix))

    def test_budget_extension_rejects_invalid_prefix_or_budget(self):
        for count, prefix in [(20, [1, 1]), (20, [5]), (1, [1, 2]), (0, [])]:
            with self.assertRaises(ValueError):
                choose_training_ids(range(100), [100], [101], [5], 42, count, prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
