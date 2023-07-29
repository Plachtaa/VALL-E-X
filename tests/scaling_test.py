# Copyright    2023                             (authors: Feiteng Li)
#

import unittest

import numpy as np
import torch
from icefall.utils import AttributeDict

from valle.models import NUM_MEL_BINS, get_model


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test_scaling_transformer(self):
        params = AttributeDict()
        params.decoder_dim = 64
        params.nhead = 4
        params.num_decoder_layers = 4

        x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
        x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
        x_lens[-1] = 8

        y = torch.from_numpy(
            np.random.random((4, 16, NUM_MEL_BINS)).astype(np.float32)
        )
        y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
        y_lens[-1] = 16

        params.model_name = "Transformer"
        params.norm_first = False
        params.add_prenet = False
        params.scaling_xformers = True

        for device in self.devices:
            # Transformer
            model = get_model(params)
            num_param = sum([p.numel() for p in model.parameters()])

            model.to(device)
            x = x.to(device)
            x_lens = x_lens.to(device)
            y = y.to(device)
            y_lens = y_lens.to(device)

            # Training
            codes, loss, metrics = model(x, x_lens, y, y_lens)
            # Inference
            model.eval()
            codes = model.inference(x[-1:], x_lens[-1:])
            params.add_prenet = False


if __name__ == "__main__":
    unittest.main()
