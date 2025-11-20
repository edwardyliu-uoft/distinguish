"""Example classification script.
Run: python examples/classify_example.py path/to/weights.pt image1.jpg image2.png
"""

import sys
from distinguish.predict import Classifier


def main():
    """Run classification on images from command line arguments."""
    if len(sys.argv) < 3:
        print(
            "Usage: python examples/classify_example.py <weights> <image1> [image2 ...]"
        )
        sys.exit(1)
    weights = sys.argv[1]
    images = sys.argv[2:]
    clf = Classifier(weights)
    results = clf.predict(images)
    for r in results:
        print(f"{r['path']}: {r['label']} (score={r['score']:.4f})")


if __name__ == "__main__":
    main()
