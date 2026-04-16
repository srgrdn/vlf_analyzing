#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_IMAGE_WIDTH = 1000
DEFAULT_IMAGE_HEIGHT = 400
DEFAULT_CLASS_LABELS = ("not_whistler", "whistler")
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "whistler_classify_hq.h5"


@dataclass
class PredictionResult:
    image_path: Path
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run whistler classification on one image or a directory of images using "
            "the trained Keras model from the notebook."
        )
    )
    parser.add_argument("input", type=Path, help="Path to an image file or a directory with images.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved visualizations and reports.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization windows.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save images or reports to disk.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when input is a directory.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Input image width for the model. Default: 1000",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Input image height for the model. Default: 400",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI. Default: 160",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions for directory scanning.",
    )
    return parser.parse_args()


def configure_matplotlib(show: bool):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def load_runtime() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not available in the current interpreter. "
            "Run this script from the same environment as the notebook, likely "
            "/mnt/c/code/vlf/.venv/Scripts/python.exe on Windows."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load images for inference.") from exc

    return tf, Image


def normalize_extensions(values: list[str]) -> set[str]:
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in values}


def collect_image_paths(input_path: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    paths = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"No images with extensions {sorted(extensions)} found in {input_path}")
    return paths


def resolve_output_dir(input_path: Path, output_dir: Path | None, save_enabled: bool) -> Path | None:
    if not save_enabled:
        return None
    if output_dir is not None:
        return output_dir.resolve()
    if input_path.is_file():
        return input_path.resolve().parent / f"{input_path.stem}_classified"
    return input_path.resolve().parent / f"{input_path.name}_classified"


def ensure_dir(path: Path | None) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def load_model(tf: Any, model_path: Path):
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def load_image_array(
    image_path: Path, image_width: int, image_height: int, image_module: Any
) -> tuple[np.ndarray, np.ndarray]:
    image = image_module.open(image_path).convert("RGB")
    image = image.resize((image_width, image_height))
    preview = np.asarray(image)
    model_input = preview.astype(np.float32)[None, ...]
    return preview, model_input


def predict_image(
    tf: Any,
    model: Any,
    image_path: Path,
    image_width: int,
    image_height: int,
    image_module: Any,
    class_labels: tuple[str, ...],
) -> PredictionResult:
    preview, model_input = load_image_array(image_path, image_width, image_height, image_module)
    raw_prediction = np.asarray(model.predict(model_input, verbose=0)).squeeze()

    if raw_prediction.ndim == 0:
        whistler_prob = float(tf.math.sigmoid(raw_prediction).numpy())
        probabilities = np.array([1.0 - whistler_prob, whistler_prob], dtype=np.float32)
    else:
        raw_prediction = np.atleast_1d(raw_prediction).astype(np.float32)
        probabilities = np.asarray(tf.nn.softmax(raw_prediction).numpy(), dtype=np.float32)

    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])
    return PredictionResult(
        image_path=image_path,
        predicted_index=predicted_index,
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
    )


def render_single_result(
    plt,
    result: PredictionResult,
    image_width: int,
    image_height: int,
    image_module: Any,
    class_labels: tuple[str, ...],
    output_path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    preview, _ = load_image_array(result.image_path, image_width, image_height, image_module)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].imshow(preview)
    axes[0].set_title(result.image_path.name)
    axes[0].axis("off")

    bars = axes[1].bar(class_labels, result.probabilities, color=["#94a3b8", "#f97316"])
    bars[result.predicted_index].set_color("#16a34a")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability")
    axes[1].set_title(
        f"Predicted: {result.predicted_label} ({result.confidence * 100:.2f}%)"
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def render_summary(
    plt,
    results: list[PredictionResult],
    class_labels: tuple[str, ...],
    output_path: Path | None,
    dpi: int,
    show: bool,
) -> None:
    counts = Counter(result.predicted_label for result in results)
    whistler_index = class_labels.index("whistler") if "whistler" in class_labels else len(class_labels) - 1
    whistler_scores = [float(result.probabilities[whistler_index]) for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].bar(class_labels, [counts.get(label, 0) for label in class_labels], color=["#94a3b8", "#f97316"])
    axes[0].set_title("Predicted Class Counts")
    axes[0].set_ylabel("Images")

    axes[1].hist(whistler_scores, bins=20, color="#f97316", alpha=0.85)
    axes[1].set_title("Whistler Probability Distribution")
    axes[1].set_xlabel("Probability")
    axes[1].set_ylabel("Images")

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def write_predictions_csv(results: list[PredictionResult], output_path: Path) -> None:
    fieldnames = ["image_path", "predicted_label", "confidence", *DEFAULT_CLASS_LABELS]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "image_path": str(result.image_path),
                "predicted_label": result.predicted_label,
                "confidence": f"{result.confidence:.6f}",
            }
            for idx, label in enumerate(DEFAULT_CLASS_LABELS):
                row[label] = f"{float(result.probabilities[idx]):.6f}"
            writer.writerow(row)


def build_stats(results: list[PredictionResult], class_labels: tuple[str, ...]) -> dict[str, Any]:
    counts = Counter(result.predicted_label for result in results)
    confidence_by_label: dict[str, list[float]] = defaultdict(list)
    for result in results:
        confidence_by_label[result.predicted_label].append(result.confidence)

    stats = {
        "total_images": len(results),
        "counts": {label: counts.get(label, 0) for label in class_labels},
        "average_confidence": {
            label: (
                float(np.mean(confidence_by_label[label])) if confidence_by_label[label] else 0.0
            )
            for label in class_labels
        },
    }
    return stats


def write_stats_json(stats: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def print_stats(stats: dict[str, Any], class_labels: tuple[str, ...]) -> None:
    print(f"Total images      : {stats['total_images']}")
    for label in class_labels:
        print(f"Count {label:12}: {stats['counts'][label]}")
    for label in class_labels:
        print(f"Avg conf {label:9}: {stats['average_confidence'][label] * 100:.2f}%")


def main() -> None:
    args = parse_args()
    if args.no_save and not args.show:
        raise ValueError("Nothing to do: use --show, or omit --no-save to save results.")

    input_path = args.input.expanduser().resolve()
    output_dir = resolve_output_dir(input_path, args.output_dir, save_enabled=not args.no_save)
    ensure_dir(output_dir)

    tf, image_module = load_runtime()
    plt = configure_matplotlib(show=args.show)
    class_labels = DEFAULT_CLASS_LABELS
    extensions = normalize_extensions(args.extensions)
    image_paths = collect_image_paths(input_path, recursive=args.recursive, extensions=extensions)
    model = load_model(tf, args.model.expanduser().resolve())

    results: list[PredictionResult] = []
    save_individual = output_dir is not None
    individual_dir = output_dir / "visualizations" if output_dir is not None and input_path.is_dir() else output_dir
    ensure_dir(individual_dir)

    for image_path in image_paths:
        result = predict_image(
            tf=tf,
            model=model,
            image_path=image_path,
            image_width=args.image_width,
            image_height=args.image_height,
            image_module=image_module,
            class_labels=class_labels,
        )
        results.append(result)

        if input_path.is_file():
            output_path = None if not save_individual else individual_dir / f"{image_path.stem}_prediction.png"
            render_single_result(
                plt=plt,
                result=result,
                image_width=args.image_width,
                image_height=args.image_height,
                image_module=image_module,
                class_labels=class_labels,
                output_path=output_path,
                dpi=args.dpi,
                show=args.show,
            )
        elif save_individual:
            output_path = individual_dir / f"{image_path.stem}_prediction.png"
            render_single_result(
                plt=plt,
                result=result,
                image_width=args.image_width,
                image_height=args.image_height,
                image_module=image_module,
                class_labels=class_labels,
                output_path=output_path,
                dpi=args.dpi,
                show=False,
            )

        print(
            f"{image_path.name}: {result.predicted_label} "
            f"({result.confidence * 100:.2f}%)"
        )

    if input_path.is_dir():
        stats = build_stats(results, class_labels)
        print_stats(stats, class_labels)

        summary_output = output_dir / "summary.png" if output_dir is not None else None
        render_summary(
            plt=plt,
            results=results,
            class_labels=class_labels,
            output_path=summary_output,
            dpi=args.dpi,
            show=args.show,
        )

        if output_dir is not None:
            write_predictions_csv(results, output_dir / "predictions.csv")
            write_stats_json(stats, output_dir / "stats.json")
            print(f"Saved summary     : {summary_output}")
            print(f"Saved CSV         : {output_dir / 'predictions.csv'}")
            print(f"Saved JSON        : {output_dir / 'stats.json'}")
    else:
        result = results[0]
        print(f"Predicted label   : {result.predicted_label}")
        print(f"Confidence        : {result.confidence * 100:.2f}%")
        if output_dir is not None:
            print(f"Saved visualization: {output_dir / f'{input_path.stem}_prediction.png'}")


if __name__ == "__main__":
    main()
