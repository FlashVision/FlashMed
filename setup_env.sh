#!/bin/bash
set -e

echo "================================================"
echo "  FlashMed Environment Setup"
echo "================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "  ✓ Python >= 3.9"
else
    echo "  ✗ Python >= 3.9 required (found $PYTHON_VERSION)"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing FlashMed with all dependencies..."
pip install -e ".[all,dev]"

echo ""
echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "Running health check..."
flashmed check

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Get started with:"
echo "  flashmed check        # Verify installation"
echo "  flashmed version      # Show version"
echo "  flashmed settings     # Show system info"
echo ""
