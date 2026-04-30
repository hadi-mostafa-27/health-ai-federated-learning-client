import unittest

import torch

from core.secure_aggregation import (
    aggregate_masked_weighted_state_dicts,
    mask_weighted_state_dict_for_upload,
)


class SecureAggregationSimulationTests(unittest.TestCase):
    def test_masked_weighted_average_matches_plain_fedavg(self):
        state_a = {"w": torch.tensor([1.0, 2.0])}
        state_b = {"w": torch.tensor([3.0, 4.0])}
        masked_a, _ = mask_weighted_state_dict_for_upload(state_a, "a", ["a", "b"], 1, 2, 0.1)
        masked_b, _ = mask_weighted_state_dict_for_upload(state_b, "b", ["a", "b"], 1, 6, 0.1)

        aggregated, metadata = aggregate_masked_weighted_state_dicts(
            {"a": masked_a, "b": masked_b},
            {"a": 2, "b": 6},
            ["a", "b"],
            1,
            0.1,
        )

        self.assertEqual(metadata["security_mode"], "secure_agg_sim")
        self.assertTrue(torch.allclose(aggregated["w"], torch.tensor([2.5, 3.5]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
