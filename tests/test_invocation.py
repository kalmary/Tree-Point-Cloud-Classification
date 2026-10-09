import ast
from pathlib import Path


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
