#!/usr/bin/env python3
"""
Quick installation check script.

Verifies that Active Torchference is properly installed and ready to use.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version compatible")
    return True


def check_package_import():
    """Check if package can be imported."""
    try:
        import active_torchference
        print(f"✓ Package installed: {active_torchference.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Cannot import package: {e}")
        print("\nTo fix: pip install -e .")
        return False


def check_core_modules():
    """Check if core modules can be imported."""
    modules = [
        "active_torchference.agent",
        "active_torchference.beliefs",
        "active_torchference.config",
        "active_torchference.environment",
        "active_torchference.free_energy",
        "active_torchference.policy",
        "active_torchference.utils",
        "active_torchference.output_manager",
        "active_torchference.orchestrators.logger",
        "active_torchference.orchestrators.visualizer",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check if key dependencies are installed."""
    dependencies = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pytest", "pytest"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: not installed")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if expected directories exist."""
    dirs = [
        "active_torchference",
        "tests",
        "examples",
        "docs",
        "validation",
    ]
    
    all_ok = True
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"✓ {d}/")
        else:
            print(f"⚠️  {d}/ not found")
            all_ok = False
    
    return all_ok


def check_python_path():
    """Display Python interpreter information."""
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if pyenv is being used
    if 'pyenv' in sys.executable:
        print("⚠️  Using pyenv-managed Python")
        print("   Make sure to install the package in this Python environment")
    elif 'homebrew' in sys.executable or '/opt/homebrew' in sys.executable:
        print("ℹ️  Using Homebrew Python")
    
    return True


def main():
    """Run all checks."""
    print("=" * 70)
    print("Active Torchference Installation Check")
    print("=" * 70)
    print()
    
    print("0. Python Environment Info")
    print("-" * 70)
    check_python_path()
    print()
    
    checks = []
    
    print("1. Checking Python Version")
    print("-" * 70)
    checks.append(check_python_version())
    print()
    
    print("2. Checking Package Installation")
    print("-" * 70)
    checks.append(check_package_import())
    print()
    
    if checks[-1]:  # Only check modules if package is installed
        print("3. Checking Core Modules")
        print("-" * 70)
        checks.append(check_core_modules())
        print()
    
    print("4. Checking Dependencies")
    print("-" * 70)
    checks.append(check_dependencies())
    print()
    
    print("5. Checking Directory Structure")
    print("-" * 70)
    checks.append(check_directories())
    print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    if all(checks):
        print("✅ All checks passed!")
        print("\nYou're ready to use Active Torchference.")
        print("\nNext steps:")
        print("  - Run tests: pytest -v")
        print("  - Run examples: python run_all_examples.py")
        print("  - Read docs: docs/QUICKSTART.md")
        return 0
    else:
        print("❌ Some checks failed.")
        print("\nPlease install the package:")
        print("  pip install -e .")
        print("\nOr use the Makefile:")
        print("  make install")
        print("\nFor more help, see TROUBLESHOOTING.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())

