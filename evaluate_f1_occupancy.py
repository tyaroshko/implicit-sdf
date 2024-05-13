import torch
import trimesh
import numpy as np
import pysdf
from sdf.network_tcnn import SDFNetwork

class OccupancyF1Calculator:
    def __init__(self, num_samples=50, std_dev=1e-2, num_points=2**18):
        self.num_samples = num_samples
        self.std_dev = std_dev
        self.num_points = num_points

    def _calculate_f1_metrics(self, gt, pred):
        TP = torch.sum((gt == 1) & (pred == 1)).item()
        FN = torch.sum((gt == 1) & (pred == 0)).item()
        FP = torch.sum((gt == 0) & (pred == 1)).item()

        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        return recall, precision, f1

    def _calculate_occupancy_f1(self, gt, pred):
        _, _, f11 = self._calculate_f1_metrics(gt, pred)
        _, _, f12 = self._calculate_f1_metrics(1 - gt, 1 - pred)
        average_f1 = (f11 + f12) / 2
        return average_f1

    def _read_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model = SDFNetwork()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def _read_mesh(self, mesh_path):
        mesh = trimesh.load_mesh(mesh_path)
        vs = mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        mesh.vertices = vs
        return mesh

    def _sample_points(self, mesh, points_type):
        if points_type == 'boundary':
            points = mesh.sample(self.num_points)
            points = np.array(points) + np.random.normal(0, self.std_dev, points.shape)
        elif points_type == 'uniform_volume':
            points = np.random.rand(self.num_points, 3) * 2 - 1
        else:
            raise ValueError("Invalid points type! Use 'boundary' or 'uniform_volume'")
        return points

    def calculate_occupancy_f1_for_samples(self, results_file="results.txt"):
        total_boundary_points_f1 = 0
        total_uniform_volume_points_f1 = 0

        with open(results_file, "a") as file:
            for sample in range(self.num_samples):
                ckpt_path = f"test_out/{sample}/checkpoints/ngp.pth.tar"
                mesh_path = f"test_task_meshes/{sample}.obj"

                model = self._read_model(ckpt_path)
                mesh = self._read_mesh(mesh_path)

                for points_type in ['boundary', 'uniform_volume']:
                    points = self._sample_points(mesh, points_type)
                    sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
                    gt_sdfs = -sdf_fn(points)[:, None].astype(np.float32)
                    gt = torch.tensor(gt_sdfs <= 0).cuda().to(torch.float32)
                    points = torch.tensor(points).cuda()
                    pred_sdfs = model(points)
                    pred = (pred_sdfs <= 0).to(torch.float32)

                    f1_score = self._calculate_occupancy_f1(gt, pred)

                    if points_type == 'boundary':
                        total_boundary_points_f1 += f1_score
                    else:
                        total_uniform_volume_points_f1 += f1_score

                    file.write(f"{sample} {points_type} points occupancy F1: {f1_score}\n")

        avg_boundary_points_f1 = total_boundary_points_f1 / self.num_samples
        avg_uniform_volume_points_f1 = total_uniform_volume_points_f1 / self.num_samples

        with open(results_file, "a") as file:
            file.write(f"Average boundary points occupancy F1: {avg_boundary_points_f1}\n")
            file.write(f"Average uniform volume points occupancy F1: {avg_uniform_volume_points_f1}\n")

occupancy_f1_calculator = OccupancyF1Calculator()
occupancy_f1_calculator.calculate_occupancy_f1_for_samples()
