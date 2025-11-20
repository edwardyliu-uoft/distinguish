"""Command Line Interface for distinguish.

Provides commands:
  distinguish --version
  distinguish --help
  distinguish classify --weights <weights_path> <image1> <image2> ... [--out <predictions.json>]
"""

from __future__ import annotations
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
    """Classify images as real photographs or AI-generated."""
    classifier = Classifier(weights, backbone=backbone)
    results = classifier.predict(list(images))
    logger.info(json.dumps(results, indent=2))
    if out:
        save_predictions_json(results, out)
        logger.info("Wrote predictions: %s", out)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
