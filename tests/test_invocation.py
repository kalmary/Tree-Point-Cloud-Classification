import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOCATION_MANIFEST = {
    "src/bdl_api.py": "diagnostic",
    "src/data_processing/downsample_npy.py": "operational",
    "src/data_processing/downsample_trees.py": "operational",
    "src/data_processing/downsample_trees_10k_pl.py": "operational",
    "src/data_processing/plot_cloud_npy.py": "diagnostic",
    "src/data_processing/remove_low_trees.py": "operational",
    "src/data_processing/swap_labels.py": "operational",
    "src/final_files/model.py": "developer",
    "src/final_files/model_en.py": "developer",
    "src/model_pipeline/eval_tree_classification.py": "operational",
    "src/model_pipeline/model.py": "developer",
    "src/model_pipeline/model_en.py": "developer",
    "src/model_pipeline/train_automated.py": "operational",
    "src/tree_classifier.py": "diagnostic",
}


def test_invocation_manifest_covers_every_main_guard():
    guarded_files = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in ast.walk(tree)
        ):
            guarded_files.add(path.relative_to(PROJECT_ROOT).as_posix())

    assert guarded_files == set(INVOCATION_MANIFEST)


@pytest.mark.parametrize(
    ("entry", "option"),
    [
        (["src/bdl_api.py"], "--latitude"),
        (["-m", "src.bdl_api"], "--latitude"),
        (["src/data_processing/downsample_npy.py"], "--source"),
        (["-m", "src.data_processing.downsample_npy"], "--source"),
        (["src/data_processing/downsample_trees.py"], "--source-path"),
        (["-m", "src.data_processing.downsample_trees"], "--source-path"),
        (["src/data_processing/downsample_trees_10k_pl.py"], "--source-path"),
        (["-m", "src.data_processing.downsample_trees_10k_pl"], "--source-path"),
        (["src/data_processing/remove_low_trees.py"], "--source"),
        (["-m", "src.data_processing.remove_low_trees"], "--source"),
        (["src/data_processing/plot_cloud_npy.py"], "--input-path"),
        (["-m", "src.data_processing.plot_cloud_npy"], "--input-path"),
        (["src/data_processing/swap_labels.py"], "--source-path"),
        (["-m", "src.data_processing.swap_labels"], "--source-path"),
        (["src/model_pipeline/eval_tree_classification.py"], "--model-name"),
        (["-m", "src.model_pipeline.eval_tree_classification"], "--model-name"),
        (["src/model_pipeline/train_automated.py"], "--model-name"),
        (["-m", "src.model_pipeline.train_automated"], "--model-name"),
        (["src/tree_classifier.py"], "--model-name"),
        (["-m", "src.tree_classifier"], "--model-name"),
    ],
)
def test_workflow_help_works_in_both_invocation_forms(entry, option, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert option in result.stdout
