import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "TreeClassifier",
        "model_pipeline.Train_Automated",
        "model_pipeline.Eval_TreeClassification",
        "model_pipeline._train_single_case",
        "model_pipeline._data_loader",
        "data_processing.downsample_trees",
        "data_processing.downsample_trees10KPL",
    ],
)
@pytest.mark.parametrize("parent_project", [False, True])
def test_workflows_import_with_their_own_utilities(module, parent_project, tmp_path):
    project = Path(__file__).resolve().parents[1]
    prefix = "src.TreeClassification.src" if parent_project else "src"
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
        ('model_pipeline/Train_Automated', '--model_name'),
        ('model_pipeline/Eval_TreeClassification', '--model_name'),
        ('data_processing/downsample_trees', '--source_path'),
        ('data_processing/downsample_trees10KPL', '--source_path'),
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
            'module = runpy.run_path(str(Path("Eval_TreeClassification.py").resolve())); '
            'assert module["bdl_species_to_model_label"](0) == 12',
        ],
        cwd=Path(__file__).resolve().parents[1] / 'src/model_pipeline',
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
