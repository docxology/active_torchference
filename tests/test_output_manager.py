"""Tests for unified output management."""

import pytest
import shutil
from pathlib import Path
from active_torchference.output_manager import OutputManager


def test_output_manager_initialization():
    """Test OutputManager initialization."""
    output_mgr = OutputManager(
        output_root="./test_output",
        experiment_name="test_exp"
    )
    
    assert output_mgr.experiment_name == "test_exp"
    assert output_mgr.output_root == Path("./test_output")
    assert output_mgr.experiment_dir == Path("./test_output/test_exp")
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_auto_name():
    """Test automatic experiment name generation."""
    output_mgr = OutputManager(output_root="./test_output")
    
    assert output_mgr.experiment_name.startswith("exp_")
    assert len(output_mgr.experiment_name) > 4
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_directory_creation():
    """Test that all subdirectories are created."""
    output_mgr = OutputManager(
        output_root="./test_output",
        experiment_name="test_dirs"
    )
    
    # Check all directories exist
    assert output_mgr.config_dir.exists()
    assert output_mgr.logs_dir.exists()
    assert output_mgr.checkpoints_dir.exists()
    assert output_mgr.visualizations_dir.exists()
    assert output_mgr.animations_dir.exists()
    assert output_mgr.data_dir.exists()
    assert output_mgr.metadata_dir.exists()
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_get_path():
    """Test getting paths for specific categories."""
    output_mgr = OutputManager(
        output_root="./test_output",
        experiment_name="test_paths"
    )
    
    config_path = output_mgr.get_path('config', 'test.json')
    assert config_path == output_mgr.config_dir / 'test.json'
    
    log_path = output_mgr.get_path('logs', 'metrics.json')
    assert log_path == output_mgr.logs_dir / 'metrics.json'
    
    checkpoint_path = output_mgr.get_path('checkpoints', 'model.pkl')
    assert checkpoint_path == output_mgr.checkpoints_dir / 'model.pkl'
    
    viz_path = output_mgr.get_path('visualizations', 'plot.png')
    assert viz_path == output_mgr.visualizations_dir / 'plot.png'
    
    anim_path = output_mgr.get_path('animations', 'anim.gif')
    assert anim_path == output_mgr.animations_dir / 'anim.gif'
    
    data_path = output_mgr.get_path('data', 'episode.json')
    assert data_path == output_mgr.data_dir / 'episode.json'
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_invalid_category():
    """Test error handling for invalid category."""
    output_mgr = OutputManager(
        output_root="./test_output",
        experiment_name="test_invalid"
    )
    
    with pytest.raises(ValueError):
        output_mgr.get_path('invalid_category', 'file.txt')
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_summary():
    """Test getting experiment summary."""
    output_mgr = OutputManager(
        output_root="./test_output",
        experiment_name="test_summary"
    )
    
    summary = output_mgr.get_experiment_summary()
    
    assert 'experiment_name' in summary
    assert 'output_root' in summary
    assert 'experiment_dir' in summary
    assert 'subdirectories' in summary
    
    assert summary['experiment_name'] == 'test_summary'
    assert len(summary['subdirectories']) == 7
    
    # Cleanup
    if output_mgr.experiment_dir.exists():
        shutil.rmtree(output_mgr.output_root)


def test_output_manager_list_experiments():
    """Test listing experiments in output root."""
    # Create multiple experiments
    exp1 = OutputManager(output_root="./test_output", experiment_name="exp1")
    exp2 = OutputManager(output_root="./test_output", experiment_name="exp2")
    
    experiments = OutputManager.list_experiments("./test_output")
    
    assert "exp1" in experiments
    assert "exp2" in experiments
    assert len(experiments) >= 2
    
    # Cleanup
    if Path("./test_output").exists():
        shutil.rmtree("./test_output")


def test_output_manager_list_experiments_empty():
    """Test listing experiments when directory doesn't exist."""
    experiments = OutputManager.list_experiments("./nonexistent_output")
    assert experiments == []

