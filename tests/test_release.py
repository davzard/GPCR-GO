import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "methods" / "model"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR))

import runepochgpu1_v1_log as training
from GNN import myGAT
from scripts.Evaluation import load_ic_vector, main as evaluate, smin_from_arrays


class GDNTests(unittest.TestCase):
    def test_regularizers_use_all_nodes_and_backpropagate(self):
        node_repr = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.5, 1.0, 1.0, 0.5],
                [1.5, 0.5, 0.5, 1.5],
            ],
            requires_grad=True,
        )
        factors = training.factorize(node_repr, num_factors=2)

        loss = (
            training.inter_factor_node_repulsion(factors)
            + training.inter_factor_space_separation(factors)
        )
        loss.backward()

        self.assertEqual(factors.shape, (3, 2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(node_repr.grad.shape, node_repr.shape)
        self.assertTrue(torch.isfinite(node_repr.grad).all())
        self.assertTrue((torch.count_nonzero(node_repr.grad, dim=1) > 0).all())


class ModelTests(unittest.TestCase):
    def test_cpu_forward_returns_shared_decoder_representation(self):
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        model = myGAT(
            graph,
            edge_dim=4,
            num_etypes=2,
            in_dims=[3],
            num_hidden=4,
            num_classes=4,
            num_layers=1,
            heads=[2, 2],
            activation=F.elu,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            alpha=0.0,
            decode="dot",
        )
        features = [torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])]
        edge_types = torch.tensor([0, 1])
        left = torch.tensor([0, 1])
        right = torch.tensor([1, 0])
        relation = torch.tensor([0, 0])

        logits, node_repr = model.forward_with_representation(
            features, edge_types, left, right, relation
        )

        self.assertFalse(next(model.parameters()).is_cuda)
        self.assertEqual(node_repr.shape, (2, 12))
        self.assertTrue(torch.allclose(logits, model.decode(node_repr, left, right, relation)))


class EvaluationTests(unittest.TestCase):
    def test_fixed_ic_order_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            ic_path = Path(directory) / "ic_dict.json"
            ic_path.write_text(json.dumps({"10": 2.0, "11": 1.0}), encoding="utf-8")
            ic_vec = load_ic_vector(ic_path, [11, 10])

        np.testing.assert_array_equal(ic_vec, np.array([1.0, 2.0], dtype=np.float32))
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.2], [0.1, 0.8]])
        self.assertEqual(smin_from_arrays(y_true, y_pred, ic_vec, thresholds=[0.5]), 0.0)
        metrics = evaluate(y_true, y_pred, ic_vec)
        self.assertEqual(metrics["smin"], 0.0)
        self.assertEqual(metrics["micro_f1"], 1.0)

    def test_missing_ic_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ic_path = Path(directory) / "ic_dict.json"
            ic_path.write_text(json.dumps({"10": 2.0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"column 1 \(node 11\)"):
                load_ic_vector(ic_path, [10, 11])


class CLITests(unittest.TestCase):
    def test_dry_run_defaults_for_all_aspects(self):
        expected = {
            "bp": (15, 15, 8, 0.1),
            "cc": (10, 15, 2, 0.1),
            "mf": (15, 15, 4, 1.0),
        }
        for aspect, defaults in expected.items():
            with self.subTest(aspect=aspect):
                parser = training.build_parser()
                args = parser.parse_args(
                    ["--dataset", f"reviewed6/{aspect}", "--dry-run"]
                )
                training.apply_paper_defaults(args, parser)
                summary = training.experiment_summary(args)
                self.assertEqual(
                    (
                        summary["neg_ratio_Pe"],
                        summary["pos_weight_Pn"],
                        summary["num_factors_M"],
                        summary["lambda_c"],
                    ),
                    defaults,
                )
                self.assertEqual(summary["lambda_i"], defaults[3])
                self.assertTrue(summary["use_unreviewed_proteins_SSL"])

    def test_removed_unreviewed_flags_are_not_accepted(self):
        parser = training.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--dataset", "reviewed6/bp", "--use-unreviewed"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--dataset", "reviewed6/bp", "--no-unreviewed"])

    def test_main_dry_run_does_not_load_data(self):
        original_argv = sys.argv
        try:
            sys.argv = [
                "runepochgpu1_v1_log.py",
                "--dataset",
                "reviewed6/bp",
                "--dry-run",
            ]
            with redirect_stdout(io.StringIO()) as output:
                training.main()
        finally:
            sys.argv = original_argv

        summary = json.loads(output.getvalue())
        self.assertEqual(summary["aspect"], "bp")


if __name__ == "__main__":
    unittest.main()
