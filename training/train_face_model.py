"""Train and evaluate the six-class face-expression transfer-learning model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset

from branches.face_model import (
    ARCHITECTURE,
    WEIGHT_SOURCE,
    build_face_model,
    make_evaluation_transform,
    make_training_transform,
    preprocessing_metadata,
)
from config import (
    FACE_CLASS_NAMES,
    FACE_DATASET_ROOT,
    FACE_DISPLAY_LABELS,
    FACE_MODEL_DIR,
)

logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NEAR_DUPLICATE_HAMMING_THRESHOLD = 4
CHECKPOINT_FORMAT_VERSION = 1


class DatasetValidationError(ValueError):
    """Raised when the dataset layout or image files are invalid."""


@dataclass(frozen=True)
class DatasetSplits:
    train: Path
    test: Path


@dataclass(frozen=True)
class ImageSample:
    path: Path
    class_index: int
    exact_hash: str
    perceptual_hash: int


class FaceImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: Sequence[ImageSample], transform: Any) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image), sample.class_index


def discover_dataset_splits(dataset_root: Path) -> DatasetSplits:
    """Find one sibling train/test pair, including a single nested archive folder."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {root}")

    candidates: set[Path] = set()
    possible_train_dirs = [root / "train", *(p for p in root.rglob("train") if p != root)]
    for train_dir in possible_train_dirs:
        if train_dir.is_dir() and (train_dir.parent / "test").is_dir():
            candidates.add(train_dir.resolve())
    if not candidates:
        raise DatasetValidationError(
            f"Expected sibling train/ and test/ directories under {root} "
            "or one of its subdirectories."
        )
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise DatasetValidationError(f"Found multiple train/test pairs; set one dataset root: {paths}")
    train_dir = next(iter(candidates))
    return DatasetSplits(train=train_dir, test=train_dir.parent / "test")


def _image_hashes(path: Path) -> tuple[str, int]:
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            pixels = np.asarray(image)
            exact_digest = hashlib.sha256(
                f"{pixels.shape[1]}x{pixels.shape[0]}:".encode("ascii") + pixels.tobytes()
            ).hexdigest()
            gray = np.asarray(image.resize((9, 8)).convert("L"), dtype=np.int16)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DatasetValidationError(f"Unreadable image {path}: {exc}") from exc

    perceptual_hash = 0
    for row in range(8):
        for column in range(8):
            perceptual_hash = (perceptual_hash << 1) | int(
                gray[row, column] > gray[row, column + 1]
            )
    return exact_digest, perceptual_hash


def scan_split(split_dir: Path) -> list[ImageSample]:
    """Validate all six class folders and index supported, readable images."""
    missing = [name for name in FACE_CLASS_NAMES if not (split_dir / name).is_dir()]
    if missing:
        raise DatasetValidationError(
            f"Missing class folder(s) in {split_dir}: {', '.join(missing)}"
        )

    samples: list[ImageSample] = []
    class_to_index = {name: index for index, name in enumerate(FACE_CLASS_NAMES)}
    for class_name in FACE_CLASS_NAMES:
        class_dir = split_dir / class_name
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not paths:
            raise DatasetValidationError(f"No supported images found in {class_dir}")
        for path in paths:
            exact_hash, perceptual_hash = _image_hashes(path)
            samples.append(
                ImageSample(
                    path=path,
                    class_index=class_to_index[class_name],
                    exact_hash=exact_hash,
                    perceptual_hash=perceptual_hash,
                )
            )
    return samples


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _related_groups(samples: Sequence[ImageSample]) -> list[list[int]]:
    """Connect exact copies and close dHash variants into indivisible split groups."""
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    exact_groups: dict[str, int] = {}
    for index, sample in enumerate(samples):
        previous = exact_groups.get(sample.exact_hash)
        if previous is None:
            exact_groups[sample.exact_hash] = index
        else:
            union(index, previous)

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if _hamming_distance(
                samples[left].perceptual_hash,
                samples[right].perceptual_hash,
            ) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        groups[find(index)].append(index)
    return list(groups.values())


def grouped_validation_indices(
    samples: Sequence[ImageSample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split whole near-duplicate groups, preserving all classes in training."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = _related_groups(samples)
    total_by_class = Counter(sample.class_index for sample in samples)
    target_by_class = {
        class_index: max(1, round(count * validation_fraction))
        for class_index, count in total_by_class.items()
    }
    validation_by_class: Counter[int] = Counter()
    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)
    validation: set[int] = set()

    while any(validation_by_class[c] < target for c, target in target_by_class.items()):
        candidates: list[tuple[float, int, list[int]]] = []
        for group in shuffled_groups:
            if all(index in validation for index in group):
                continue
            group_counts = Counter(samples[index].class_index for index in group)
            if any(
                total_by_class[class_index] - validation_by_class[class_index] - count < 1
                for class_index, count in group_counts.items()
            ):
                continue
            gain = sum(
                min(count, max(0, target_by_class[class_index] - validation_by_class[class_index]))
                / target_by_class[class_index]
                for class_index, count in group_counts.items()
            )
            if gain > 0:
                candidates.append((gain / len(group), gain, group))
        if not candidates:
            break
        _, _, chosen = max(candidates, key=lambda item: (item[0], item[1]))
        validation.update(chosen)
        validation_by_class.update(samples[index].class_index for index in chosen)

    missing_classes = [
        FACE_CLASS_NAMES[class_index]
        for class_index in target_by_class
        if validation_by_class[class_index] == 0
    ]
    if missing_classes:
        raise DatasetValidationError(
            "Unable to include every class in grouped validation without leakage: "
            + ", ".join(missing_classes)
        )
    train_indices = [index for index in range(len(samples)) if index not in validation]
    validation_indices = sorted(validation)
    return train_indices, validation_indices


def audit_train_test_overlap(
    train_samples: Sequence[ImageSample],
    test_samples: Sequence[ImageSample],
) -> dict[str, Any]:
    """Screen train/test for exact and close perceptual duplicates."""
    exact_pairs = [
        (train.path.name, test.path.name)
        for train in train_samples
        for test in test_samples
        if train.exact_hash == test.exact_hash
    ]
    near_pairs = []
    for train in train_samples:
        for test in test_samples:
            if train.exact_hash != test.exact_hash and _hamming_distance(
                train.perceptual_hash, test.perceptual_hash
            ) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                near_pairs.append((train.path.name, test.path.name))
    return {
        "exact_duplicate_pairs": len(exact_pairs),
        "near_duplicate_candidate_pairs_dhash_le_4": len(near_pairs),
        "examples": [list(pair) for pair in (exact_pairs + near_pairs)[:25]],
        "warning": (
            "Perceptual-hash matches are screening candidates, not verified identities. "
            "The supplied test set remains untouched; possible train/test related images "
            "mean test metrics are not an independent generalization estimate."
            if exact_pairs or near_pairs
            else None
        ),
    }


def audit_training_duplicates(samples: Sequence[ImageSample]) -> dict[str, int]:
    """Count exact duplicates and exact duplicate groups with conflicting labels."""
    groups: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.exact_hash].append(sample)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    return {
        "exact_duplicate_groups": len(duplicate_groups),
        "extra_exact_duplicate_files": sum(len(group) - 1 for group in duplicate_groups),
        "cross_label_exact_duplicate_groups": sum(
            len({sample.class_index for sample in group}) > 1
            for group in duplicate_groups
        ),
    }


class IndexedDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, base: FaceImageDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.base[self.indices[index]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    if training:
        model.features.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    labels: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            count = targets.size(0)
            total_loss += float(loss.item()) * count
            total_correct += int((predicted == targets).sum().item())
            total_count += count
            labels.extend(int(label) for label in targets.cpu().tolist())
            predictions.extend(int(label) for label in predicted.cpu().tolist())
    if total_count == 0:
        raise RuntimeError("Cannot evaluate an empty dataset.")
    return total_loss / total_count, total_correct / total_count, labels, predictions


def calculate_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, Any]:
    """Compute class-wise precision, recall, F1, and true-row/predicted-column matrix."""
    class_count = len(FACE_CLASS_NAMES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        confusion[int(actual), int(predicted)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(FACE_CLASS_NAMES):
        true_positive = int(confusion[index, index])
        predicted_count = int(confusion[:, index].sum())
        support = int(confusion[index, :].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_rows_true_columns_predicted": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _check_output_files(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        "model.pt",
        "metadata.json",
        "class_to_index.json",
        "preprocessing.json",
        "training_results.json",
    )
    existing = [output_dir / name for name in outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing model output(s): "
            f"{', '.join(str(path) for path in existing)}. "
            "Choose a new --output-dir or explicitly pass --overwrite."
        )


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    _check_output_files(output_dir, args.overwrite)
    _seed_everything(args.seed)
    device = _select_device(args.device)
    splits = discover_dataset_splits(args.data_root)
    train_samples = scan_split(splits.train)
    test_samples = scan_split(splits.test)
    training_duplicate_audit = audit_training_duplicates(train_samples)
    train_indices, validation_indices = grouped_validation_indices(
        train_samples,
        args.validation_fraction,
        args.seed,
    )
    overlap_audit = audit_train_test_overlap(train_samples, test_samples)
    if overlap_audit["warning"]:
        logger.warning("%s", overlap_audit["warning"])
        logger.warning(
            "Potential train/test overlap: exact=%d, near=%d",
            overlap_audit["exact_duplicate_pairs"],
            overlap_audit["near_duplicate_candidate_pairs_dhash_le_4"],
        )

    class_to_index = {name: index for index, name in enumerate(FACE_CLASS_NAMES)}
    transforms_by_split = {
        "train": make_training_transform(),
        "eval": make_evaluation_transform(),
    }
    base_train = FaceImageDataset(train_samples, transforms_by_split["train"])
    eval_train = FaceImageDataset(train_samples, transforms_by_split["eval"])
    test_dataset = FaceImageDataset(test_samples, transforms_by_split["eval"])
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        IndexedDataset(base_train, train_indices),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        IndexedDataset(eval_train, validation_indices),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    try:
        model = build_face_model(num_classes=len(FACE_CLASS_NAMES), pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize MobileNetV3-Small with pretrained ImageNet weights. "
            f"Check network access to {WEIGHT_SOURCE}. No from-scratch fallback was used."
        ) from exc
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = _run_epoch(
            model, validation_loader, criterion, device, None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        logger.info(
            "Epoch %d/%d train loss %.4f acc %.3f; validation loss %.4f acc %.3f",
            epoch,
            args.epochs,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    _, _, test_labels, test_predictions = _run_epoch(
        model, test_loader, criterion, device, None
    )
    test_metrics = calculate_metrics(test_labels, test_predictions)
    test_metrics["independent_evaluation_warning"] = overlap_audit["warning"]
    test_metrics["train_test_overlap_audit"] = overlap_audit

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "class_names": list(FACE_CLASS_NAMES),
        "class_to_index": class_to_index,
        "display_label_mapping": FACE_DISPLAY_LABELS,
        "preprocessing": preprocessing_metadata(),
        "pretrained_weight_source": WEIGHT_SOURCE,
        "pretrained_weight_enum": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "backbone_frozen": True,
        "trained_parameters": "classification head only",
        "device": str(device),
        "seed": args.seed,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
        },
        "dataset": {
            "configured_root": str(Path(args.data_root).resolve()),
            "actual_train_dir": str(splits.train),
            "actual_test_dir": str(splits.test),
            "train_image_count": len(train_samples),
            "train_subset_count": len(train_indices),
            "validation_count": len(validation_indices),
            "test_image_count": len(test_samples),
            "training_duplicate_audit": training_duplicate_audit,
            "validation_split_method": (
                "seeded group holdout; exact-pixel duplicates and 64-bit dHash "
                f"pairs with Hamming distance <= {NEAR_DUPLICATE_HAMMING_THRESHOLD} "
                "are assigned to the same group"
            ),
        },
        "test_metrics_independent": not bool(overlap_audit["warning"]),
    }
    checkpoint = {
        "metadata": metadata,
        "state_dict": best_state,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "class_to_index.json", class_to_index)
    _write_json(output_dir / "preprocessing.json", metadata["preprocessing"])
    _write_json(
        output_dir / "training_results.json",
        {
            "best_validation_accuracy": best_validation_accuracy,
            "history": history,
            "test_metrics": test_metrics,
        },
    )
    logger.info("Saved model checkpoint and metadata to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "best_validation_accuracy": best_validation_accuracy,
        "test_metrics": test_metrics,
        "history": history,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FACE_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FACE_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing model, metadata, and result files in the output directory",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("epochs and batch-size must be positive; num-workers cannot be negative.")
    result = train(args)
    print(json.dumps(
        {
            "output_dir": result["output_dir"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "test_metrics": result["test_metrics"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
