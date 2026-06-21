"""Example: Whole slide image analysis for computational pathology."""

from flashmed import FlashMed, Trainer
from flashmed.solutions.pathology_analyzer import PathologyAnalyzer
from flashmed.tasks.pathology import PathologyTask


def train_pathology_classifier():
    """Train on PathMNIST dataset for tissue classification."""
    trainer = Trainer(
        task="pathology",
        data_dir="data/pathmnist",
        num_classes=9,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        device="cuda",
        save_dir="workspace/pathology",
    )
    trainer.train()


def analyze_slide():
    """Analyze a whole slide image."""
    analyzer = PathologyAnalyzer(
        model_path="workspace/pathology/flashmed_best.pth",
        device="cuda",
        patch_size=256,
        batch_size=32,
    )

    result = analyzer.analyze("data/slides/sample_slide.png", generate_heatmap=True)

    print(f"Slide: {result['slide_path']}")
    print(f"Patches analyzed: {result['num_patches']}")
    print(f"Prediction: {result['slide_prediction']['predicted_label']}")
    print(f"Confidence: {result['slide_prediction']['mean_confidence']:.2%}")


def extract_and_classify_patches():
    """Demonstrate patch extraction from a large image."""
    import numpy as np
    from PIL import Image

    task = PathologyTask(num_classes=9, patch_size=256)

    image = np.random.randint(50, 200, (2048, 2048, 3), dtype=np.uint8)
    patches = task.extract_patches(image, overlap=0.25)
    print(f"Extracted {len(patches)} tissue patches from image")


if __name__ == "__main__":
    train_pathology_classifier()
