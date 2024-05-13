import torch
import numpy as np
from sdf.network_tcnn import SDFNetwork

class ModelTimer:
    def __init__(self, model_path, input_size=(1000000, 3)):
        self.model = self._load_model(model_path)
        self.dummy_input = torch.randn(*input_size, dtype=torch.float).cuda()
        self.timer = self._initialize_timer()

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = SDFNetwork()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def _initialize_timer(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        return starter, ender

    def measure_inference_time(self, repetitions=300):
        timings = np.zeros((repetitions, 1))

        # Perform warm-up runs
        for _ in range(10):
            _ = self.model(self.dummy_input)

        # Measure time for each repetition
        with torch.no_grad():
            for rep in range(repetitions):
                self.timer[0].record()
                _ = self.model(self.dummy_input)
                self.timer[1].record()
                torch.cuda.synchronize()
                timings[rep] = self.timer[0].elapsed_time(self.timer[1])

        return np.min(timings)

    def measure_single_point_inference_time(self, repetitions=300):
        timings = np.zeros((repetitions, 1))

        single_point_input = torch.randn(1, 3, dtype=torch.float).cuda()

        # Perform warm-up runs
        for _ in range(10):
            _ = self.model(single_point_input)

        # Measure time for each repetition
        with torch.no_grad():
            for rep in range(repetitions):
                self.timer[0].record()
                _ = self.model(single_point_input)
                self.timer[1].record()
                torch.cuda.synchronize()
                timings[rep] = self.timer[0].elapsed_time(self.timer[1])

        return np.min(timings) * 1e6

model_path = "test_out/0/checkpoints/ngp.pth.tar"
timer = ModelTimer(model_path)
min_time = timer.measure_inference_time()
print("Minimum time for 100k points:", min_time, "milliseconds")
min_single_point_time = timer.measure_single_point_inference_time()
print("Minimum time for a single point:", min_single_point_time, "nanoseconds")
