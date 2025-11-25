"""Command Line Interface for distinguish.

Provides commands:
  distinguish --version
  distinguish --help
  distinguish classify --weights <weights_path> <image1> <image2> ... [--out <predictions.json>]
"""

from __future__ import annotations
from pathlib import Path
import logging
import json

import click

from .predict import Classifier, save_predictions_json

logger = logging.getLogger("distinguish")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__)
def cli():
    """Binary classifier distinguishing real photographs from AI-generated images."""


@cli.command()
@click.option("--weights", required=True, help="Path to trained model weights")
@click.option("--out", help="Output JSON path for predictions")
@click.option(
    "--backbone",
    default="resnet18",
    help="Backbone architecture (resnet18, resnet34, efficientnet_b0)",
)
@click.argument("images", nargs=-1, required=True)
def classify(weights: str, out: str | None, backbone: str, images: tuple) -> None:
    """Classify images as real photographs or AI-generated.

    IMAGES can be individual image paths or directories containing images.
    Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    # Collect all image paths from arguments (files or directories)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    for path_str in images:
        path = Path(path_str)
        if path.is_file():
            # Add file directly
            image_paths.append(str(path))
        elif path.is_dir():
            # Find all image files in directory
            for ext in image_extensions:
                image_paths.extend([str(p) for p in path.glob(f"*{ext}")])
                image_paths.extend([str(p) for p in path.glob(f"*{ext.upper()}")])
        else:
            logger.warning("Path not found or not accessible: %s", path_str)

    if not image_paths:
        logger.error("No valid image files found in provided paths")
        return

    logger.info("Found %d image(s) to classify", len(image_paths))

    classifier = Classifier(weights, backbone=backbone)
    results = classifier.predict(
        image_paths,
        labelmap={1: "real", 0: "ai_generated"} if "_alt" in weights else None,
    )
    logger.info(json.dumps(results, indent=2))
    if out:
        save_predictions_json(results, out)
        logger.info("Wrote predictions: %s", out)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
