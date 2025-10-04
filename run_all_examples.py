#!/usr/bin/env python3
"""
Run all examples with comprehensive validation and logging.

Executes each example in safe-to-fail mode, validates outputs,
and generates detailed report of all saved files and analyses.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time


class ExampleValidator:
    """Validates example execution and output generation."""
    
    def __init__(self, output_root: Path = Path("./output")):
        """
        Initialize validator.
        
        Args:
            output_root: Root directory for all outputs.
        """
        self.output_root = output_root
        self.results = {}
        self.start_time = datetime.now()
    
    def run_example(self, example_path: Path, timeout: int = 300) -> Tuple[bool, str, float]:
        """
        Run a single example script.
        
        Args:
            example_path: Path to example script.
            timeout: Maximum execution time in seconds.
        
        Returns:
            Tuple of (success, output, execution_time).
        """
        print(f"\n{'='*70}")
        print(f"Running: {example_path.name}")
        print(f"{'='*70}")
        
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=example_path.parent.parent
            )
            
            execution_time = time.time() - start
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                print(f"✓ SUCCESS ({execution_time:.1f}s)")
            else:
                print(f"✗ FAILED ({execution_time:.1f}s)")
                print(f"Return code: {result.returncode}")
            
            return success, output, execution_time
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start
            print(f"✗ TIMEOUT after {timeout}s")
            return False, f"Timeout after {timeout}s", execution_time
        
        except Exception as e:
            execution_time = time.time() - start
            print(f"✗ ERROR: {e}")
            return False, str(e), execution_time
    
    def validate_output_structure(self, experiment_name: str) -> Dict:
        """
        Validate output directory structure and contents.
        
        Args:
            experiment_name: Name of experiment to validate.
        
        Returns:
            Dictionary with validation results.
        """
        experiment_dir = self.output_root / experiment_name
        
        if not experiment_dir.exists():
            return {
                "exists": False,
                "error": f"Directory not found: {experiment_dir}"
            }
        
        # Expected subdirectories
        subdirs = {
            "config": experiment_dir / "config",
            "logs": experiment_dir / "logs",
            "checkpoints": experiment_dir / "checkpoints",
            "visualizations": experiment_dir / "visualizations",
            "animations": experiment_dir / "animations",
            "data": experiment_dir / "data",
            "metadata": experiment_dir / "metadata"
        }
        
        validation = {
            "exists": True,
            "experiment_dir": str(experiment_dir),
            "subdirectories": {},
            "files": {},
            "total_files": 0
        }
        
        # Check subdirectories and count files
        for name, path in subdirs.items():
            if path.exists():
                files = list(path.glob("*"))
                validation["subdirectories"][name] = {
                    "exists": True,
                    "file_count": len(files),
                    "files": [f.name for f in files]
                }
                validation["total_files"] += len(files)
            else:
                validation["subdirectories"][name] = {
                    "exists": False,
                    "file_count": 0,
                    "files": []
                }
        
        # Load and validate specific files
        config_file = subdirs["config"] / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                validation["files"]["config"] = {
                    "exists": True,
                    "valid": True,
                    "keys": list(config.keys())
                }
            except Exception as e:
                validation["files"]["config"] = {
                    "exists": True,
                    "valid": False,
                    "error": str(e)
                }
        
        metrics_file = subdirs["logs"] / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                validation["files"]["metrics"] = {
                    "exists": True,
                    "valid": True,
                    "timesteps": len(metrics.get("timesteps", [])),
                    "vfe_values": len(metrics.get("vfe", [])),
                    "efe_values": len(metrics.get("efe", [])),
                    "keys": list(metrics.keys())
                }
            except Exception as e:
                validation["files"]["metrics"] = {
                    "exists": True,
                    "valid": False,
                    "error": str(e)
                }
        
        return validation
    
    def validate_comparative_output(self, base_dir: Path) -> Dict:
        """
        Validate comparative experiment output structure.
        
        Args:
            base_dir: Base directory containing multiple experiments.
        
        Returns:
            Dictionary with validation results.
        """
        if not base_dir.exists():
            return {
                "exists": False,
                "error": f"Directory not found: {base_dir}"
            }
        
        validation = {
            "exists": True,
            "base_dir": str(base_dir),
            "experiments": {},
            "comparison_files": []
        }
        
        # Find comparison summary files
        for item in base_dir.glob("*.png"):
            validation["comparison_files"].append(item.name)
        
        # Find sub-experiments
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("balance_"):
                exp_validation = self.validate_output_structure_relative(subdir)
                validation["experiments"][subdir.name] = exp_validation
        
        return validation
    
    def validate_output_structure_relative(self, experiment_dir: Path) -> Dict:
        """
        Validate output structure for a directory (not in standard output root).
        
        Args:
            experiment_dir: Path to experiment directory.
        
        Returns:
            Dictionary with validation results.
        """
        subdirs = {
            "config": experiment_dir / "config",
            "logs": experiment_dir / "logs",
            "checkpoints": experiment_dir / "checkpoints",
            "visualizations": experiment_dir / "visualizations",
            "animations": experiment_dir / "animations",
            "data": experiment_dir / "data",
            "metadata": experiment_dir / "metadata"
        }
        
        validation = {
            "subdirectories": {},
            "total_files": 0
        }
        
        for name, path in subdirs.items():
            if path.exists():
                files = list(path.glob("*"))
                validation["subdirectories"][name] = {
                    "exists": True,
                    "file_count": len(files)
                }
                validation["total_files"] += len(files)
            else:
                validation["subdirectories"][name] = {
                    "exists": False,
                    "file_count": 0
                }
        
        return validation
    
    def generate_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns:
            Formatted report string.
        """
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        report = []
        report.append("="*80)
        report.append("EXAMPLE VALIDATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total execution time: {total_time:.1f}s")
        report.append("")
        
        # Summary statistics
        total_examples = len(self.results)
        successful = sum(1 for r in self.results.values() if r["success"])
        failed = total_examples - successful
        
        report.append("SUMMARY")
        report.append("-"*80)
        report.append(f"Total examples: {total_examples}")
        report.append(f"Successful: {successful}")
        report.append(f"Failed: {failed}")
        report.append(f"Success rate: {successful/total_examples*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-"*80)
        
        for example_name, result in self.results.items():
            report.append(f"\n{example_name}")
            report.append("  " + "-"*76)
            
            if result["success"]:
                report.append(f"  Status: ✓ SUCCESS")
            else:
                report.append(f"  Status: ✗ FAILED")
            
            report.append(f"  Execution time: {result['execution_time']:.2f}s")
            
            if "validation" in result and result["validation"].get("exists"):
                val = result["validation"]
                report.append(f"  Output directory: {val['experiment_dir']}")
                report.append(f"  Total files: {val['total_files']}")
                
                report.append(f"\n  Subdirectories:")
                for subdir, info in val["subdirectories"].items():
                    status = "✓" if info["exists"] else "✗"
                    report.append(f"    {status} {subdir:15s} ({info['file_count']} files)")
                    if info["exists"] and info["file_count"] > 0:
                        for filename in sorted(info["files"])[:5]:
                            report.append(f"        • {filename}")
                        if info["file_count"] > 5:
                            report.append(f"        • ... and {info['file_count']-5} more")
                
                if "files" in val:
                    report.append(f"\n  Key files:")
                    for file_type, file_info in val["files"].items():
                        if file_info.get("valid"):
                            report.append(f"    ✓ {file_type}.json - Valid")
                            if file_type == "metrics":
                                report.append(f"        Timesteps: {file_info['timesteps']}")
                                report.append(f"        VFE values: {file_info['vfe_values']}")
                                report.append(f"        EFE values: {file_info['efe_values']}")
                        else:
                            report.append(f"    ✗ {file_type}.json - Invalid or missing")
            
            elif "comparative_validation" in result:
                val = result["comparative_validation"]
                report.append(f"  Output directory: {val['base_dir']}")
                report.append(f"  Comparison files: {len(val['comparison_files'])}")
                for cf in val["comparison_files"]:
                    report.append(f"    • {cf}")
                report.append(f"  Sub-experiments: {len(val['experiments'])}")
                for exp_name, exp_val in val["experiments"].items():
                    report.append(f"    • {exp_name}: {exp_val['total_files']} files")
            
            if not result["success"] and "output" in result:
                report.append(f"\n  Error output (last 200 chars):")
                error_snippet = result["output"][-200:]
                for line in error_snippet.split('\n'):
                    if line.strip():
                        report.append(f"    {line}")
        
        report.append("\n" + "="*80)
        report.append("VALIDATION COMPLETE")
        report.append("="*80)
        
        if failed == 0:
            report.append("✅ ALL EXAMPLES PASSED")
        else:
            report.append(f"⚠️  {failed} EXAMPLE(S) FAILED")
        
        report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filepath: Path):
        """
        Save report to file.
        
        Args:
            filepath: Path to save report.
        """
        report = self.generate_report()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {filepath}")


def main():
    """Run all examples with validation."""
    
    print("="*80)
    print("RUNNING ALL ACTIVE TORCHFERENCE EXAMPLES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print("")
    
    # Initialize validator
    validator = ExampleValidator()
    
    # Define examples to run
    examples_dir = Path("examples")
    examples = [
        {
            "name": "unified_output_example",
            "path": examples_dir / "unified_output_example.py",
            "output_name": "unified_demo",
            "timeout": 120
        },
        {
            "name": "simple_navigation",
            "path": examples_dir / "simple_navigation.py",
            "output_name": "simple_navigation",
            "timeout": 180
        },
        {
            "name": "gridworld_exploration",
            "path": examples_dir / "gridworld_exploration.py",
            "output_name": "gridworld_exploration",
            "timeout": 120
        },
        {
            "name": "custom_environment",
            "path": examples_dir / "custom_environment.py",
            "output_name": "oscillator_tracking",
            "timeout": 240
        },
        {
            "name": "epistemic_pragmatic_balance",
            "path": examples_dir / "epistemic_pragmatic_balance.py",
            "output_name": "epistemic_pragmatic_comparison",
            "timeout": 300,
            "comparative": True
        }
    ]
    
    # Run each example
    for example in examples:
        try:
            # Run example
            success, output, exec_time = validator.run_example(
                example["path"],
                example["timeout"]
            )
            
            # Store basic results
            validator.results[example["name"]] = {
                "success": success,
                "execution_time": exec_time,
                "output": output
            }
            
            # Validate outputs
            if success:
                print(f"\nValidating outputs for {example['name']}...")
                
                if example.get("comparative"):
                    # Comparative experiment validation
                    validation = validator.validate_comparative_output(
                        Path("output") / example["output_name"]
                    )
                    validator.results[example["name"]]["comparative_validation"] = validation
                    
                    if validation.get("exists"):
                        print(f"  ✓ Output directory exists")
                        print(f"  ✓ Comparison files: {len(validation['comparison_files'])}")
                        print(f"  ✓ Sub-experiments: {len(validation['experiments'])}")
                    else:
                        print(f"  ✗ Output directory not found")
                else:
                    # Standard experiment validation
                    validation = validator.validate_output_structure(example["output_name"])
                    validator.results[example["name"]]["validation"] = validation
                    
                    if validation.get("exists"):
                        print(f"  ✓ Output directory exists")
                        print(f"  ✓ Total files: {validation['total_files']}")
                        
                        # Check subdirectories
                        for subdir, info in validation["subdirectories"].items():
                            if info["exists"]:
                                print(f"  ✓ {subdir}: {info['file_count']} files")
                            else:
                                print(f"  ✗ {subdir}: missing")
                    else:
                        print(f"  ✗ Output directory not found")
            else:
                print(f"\n⚠️  Skipping validation due to execution failure")
        
        except Exception as e:
            print(f"\n✗ EXCEPTION: {e}")
            validator.results[example["name"]] = {
                "success": False,
                "execution_time": 0,
                "output": str(e)
            }
    
    # Generate and display report
    print("\n")
    report = validator.generate_report()
    print(report)
    
    # Save report
    report_path = Path("validation") / "outputs" / f"examples_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    validator.save_report(report_path)
    
    # Save JSON summary
    json_path = report_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(validator.results),
            "successful": sum(1 for r in validator.results.values() if r["success"]),
            "results": validator.results
        }, f, indent=2, default=str)
    
    print(f"✓ JSON summary saved to: {json_path}")
    
    # Exit with appropriate code
    all_passed = all(r["success"] for r in validator.results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

