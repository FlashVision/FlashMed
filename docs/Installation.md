# Installation

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA GPU recommended (training)

## Install from PyPI

```bash
pip install flashmed
```

## Install from Source

```bash
git clone https://github.com/FlashVision/FlashMed.git
cd FlashMed
pip install -e ".[all]"
```

## Optional Dependencies

```bash
# Privacy features (federated learning, DP)
pip install flashmed[privacy]

# Analytics and visualization
pip install flashmed[analytics]

# ONNX export
pip install flashmed[export]

# Everything
pip install flashmed[all]

# Development
pip install flashmed[dev]
```

## Automated Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
```

## Verify Installation

```bash
flashmed check
```

## Docker

```bash
cd docker
docker compose build
docker compose run flashmed check
```

## Troubleshooting

- **CUDA not found**: Install PyTorch with CUDA from https://pytorch.org
- **pydicom errors**: `pip install pydicom`
- **MONAI issues**: `pip install monai`
