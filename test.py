import unittest

import torch

from models.dcae import DCAE


class TestDCAE(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DCAE().to(self.device)
        self.model.eval()

    def test_forward_shape(self):
        print(f"\nTesting Forward pass on {self.device}...")
        x = torch.randn(1, 3, 256, 256).to(self.device)

        with torch.no_grad():
            out = self.model(x)

        self.assertIn("x_hat", out)
        self.assertEqual(
            out["x_hat"].shape,
            x.shape,
            f"Output shape mismatch. Expected {x.shape}, got {out['x_hat'].shape}",
        )

        self.assertIn("likelihoods", out)
        self.assertIn("y", out["likelihoods"])
        self.assertIn("z", out["likelihoods"])
        print("Forward pass successful.")

    def test_compress_decompress_cycle(self):
        print(f"\nTesting Compress/Decompress cycle on {self.device}...")
        height, width = 256, 256
        x = torch.rand(1, 3, height, width).to(self.device)

        with torch.no_grad():
            compressed_out = self.model.compress(x)

        self.assertIn("strings", compressed_out)
        self.assertIn("shape", compressed_out)

        strings = compressed_out["strings"]
        shape = compressed_out["shape"]

        self.assertEqual(
            len(strings), 2, "Compressed output should contain strings for y and z"
        )

        with torch.no_grad():
            decompressed_out = self.model.decompress(strings, shape)

        self.assertIn("x_hat", decompressed_out)
        x_hat = decompressed_out["x_hat"]

        self.assertEqual(
            x_hat.shape,
            x.shape,
            f"Decompressed shape mismatch. Expected {x.shape}, got {x_hat.shape}",
        )

        self.assertTrue(
            x_hat.min() >= 0 and x_hat.max() <= 1,
            "Output image values out of range [0, 1]",
        )

        print(
            f"Compress/Decompress cycle successful. Input: {x.shape} -> Output: {x_hat.shape}"
        )

    def test_variable_resolution(self):
        print(f"\nTesting Variable Resolution (128x128)...")
        x = torch.rand(1, 3, 128, 128).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out["x_hat"].shape, x.shape)
        print("Variable resolution test successful.")


if __name__ == "__main__":
    unittest.main()
