import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "tree_classifier",
        "model_pipeline.train_automated",
        "model_pipeline.eval_tree_classification",
        "model_pipeline._train_single_case",
        "model_pipeline._data_loader",
        "data_processing.downsample_trees",
        "data_processing.downsample_trees_10k_pl",
    ],
)
@pytest.mark.parametrize("parent_project", [False, True])
def test_workflows_import_with_their_own_utilities(module, parent_project, tmp_path):
    project = Path(__file__).resolve().parents[1]
    prefix = "src.tree_classification.src" if parent_project else "src"
    code = f"""
import importlib

module = importlib.import_module('{prefix}.{module}')
utilities = importlib.import_module('{prefix}.utils.nn_utils')
for name in ('load_json', 'load_model', 'compute_pos_weights', 'FocalLoss', 'Plotter'):
    if name in vars(module):
        assert getattr(module, name) is getattr(utilities, name), name
if hasattr(module, 'bdl_species_to_model_label'):
    assert module.bdl_species_to_model_label(0) == 12
"""
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=project.parents[1] if parent_project else project,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ('script', 'option'),
    [
        ('model_pipeline/train_automated', '--model-name'),
        ('model_pipeline/eval_tree_classification', '--model-name'),
        ('data_processing/downsample_trees', '--source-path'),
        ('data_processing/downsample_trees_10k_pl', '--source-path'),
    ],
)
@pytest.mark.parametrize('module_execution', [False, True])
def test_workflow_help_preserves_both_invocations(script, option, module_execution, tmp_path):
    entry = ['-m', 'src.' + script.replace('/', '.')] if module_execution else [f'src/{script}.py']
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    result = subprocess.run(
        [sys.executable, *entry, '--help'],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert option in result.stdout


def test_direct_evaluation_preserves_species_metadata_import():
    result = subprocess.run(
        [
            sys.executable, '-c',
            'import runpy; from pathlib import Path; '
            'module = runpy.run_path(str(Path("eval_tree_classification.py").resolve())); '
            'assert module["bdl_species_to_model_label"](0) == 12',
        ],
        cwd=Path(__file__).resolve().parents[1] / 'src/model_pipeline',
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_tree_classifier_import_does_not_load_offline_utilities():
    code = """
import importlib.abc
import sys

class BlockOfflineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'fpsample', 'h5py', 'laspy'
        }:
            raise ImportError(f'Offline dependency imported: {fullname}')

sys.meta_path.insert(0, BlockOfflineImports())
from src.tree_classifier import TreeClassifier

forbidden = {
    'fpsample',
    'h5py',
    'laspy',
    'matplotlib',
    'optuna',
    'pandas',
    'pyvista',
    'seaborn',
    'sklearn',
    'torchinfo',
    'src.utils.pcd_manipulation',
    'src.utils.nn_utils.src.accuracy_metrics',
    'src.utils.nn_utils.src.evaluation_plot_tools',
    'src.utils.nn_utils.src.loss_functions',
    'src.utils.nn_utils.src.training_callbacks',
}
loaded = forbidden & sys.modules.keys()
assert not loaded, loaded
assert TreeClassifier.__module__ == 'src.tree_classifier'
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
