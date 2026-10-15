import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from config import FACE_CLASS_NAMES
from training.train_face_model import (
    DatasetValidationError,
    ImageSample,
    audit_train_test_overlap,
    audit_training_duplicates,
    calculate_metrics,
    discover_dataset_splits,
    grouped_validation_indices,
    scan_split,
    _check_output_files,
)


def write_class_images(split: Path, copies: int = 2) -> None:
    rng = np.random.default_rng(314)
    for class_index, class_name in enumerate(FACE_CLASS_NAMES):
        folder = split / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for image_index in range(copies):
            pixels = rng.integers(
                0,
                256,
                size=(80, 80, 3),
                dtype=np.uint8,
            )
            Image.fromarray(pixels).save(folder / f"{image_index}.jpg")


class FaceTrainingDataTests(unittest.TestCase):
    def test_training_does_not_overwrite_existing_outputs_without_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "model.pt").write_bytes(b"existing")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                _check_output_files(output_dir, overwrite=False)
            _check_output_files(output_dir, overwrite=True)

    def test_finds_nested_train_and_test_and_scans_all_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "archive" / "dataset"
            train, test = nested / "train", nested / "test"
            write_class_images(train)
            write_class_images(test, copies=1)

            splits = discover_dataset_splits(root)
            train_samples = scan_split(splits.train)
            test_samples = scan_split(splits.test)

        self.assertEqual(splits.train, nested / "train")
        self.assertEqual(len(train_samples), 12)
        self.assertEqual(len(test_samples), 6)
        self.assertEqual(
            {sample.class_index for sample in train_samples},
            set(range(len(FACE_CLASS_NAMES))),
        )

    def test_rejects_missing_dataset_path_and_missing_class_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(DatasetValidationError):
                discover_dataset_splits(root / "missing")

            split = root / "train"
            split.mkdir()
            for class_name in FACE_CLASS_NAMES[:-1]:
                (split / class_name).mkdir()
            with self.assertRaisesRegex(DatasetValidationError, "surprise"):
                scan_split(split)

    def test_reports_unreadable_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "train"
            for class_name in FACE_CLASS_NAMES:
                (split / class_name).mkdir(parents=True)
            (split / FACE_CLASS_NAMES[0] / "broken.jpg").write_bytes(b"not an image")

            with self.assertRaisesRegex(DatasetValidationError, "Unreadable image"):
                scan_split(split)

    def test_related_images_are_kept_in_one_validation_group(self) -> None:
        samples = []
        distinct_hashes = [255 << (index * 8) for index in range(len(FACE_CLASS_NAMES))]
        for class_index, perceptual_hash in enumerate(distinct_hashes):
            samples.append(
                ImageSample(
                    path=Path(f"class-{class_index}-original.jpg"),
                    class_index=class_index,
                    exact_hash=f"hash-{class_index}",
                    perceptual_hash=perceptual_hash,
                )
            )
            samples.append(
                ImageSample(
                    path=Path(f"class-{class_index}-other.jpg"),
                    class_index=class_index,
                    exact_hash=f"hash-{class_index}-other",
                    perceptual_hash=perceptual_hash ^ (
                        (1 << (class_index * 8))
                        | (1 << (class_index * 8 + 1))
                        | (1 << (class_index * 8 + 2))
                    ),
                )
            )
            samples.append(
                ImageSample(
                    path=Path(f"class-{class_index}-separate.jpg"),
                    class_index=class_index,
                    exact_hash=f"hash-{class_index}-separate",
                    perceptual_hash=perceptual_hash ^ (255 << (class_index * 8)),
                )
            )
        duplicate = samples[0]
        samples.append(
            ImageSample(
                path=Path("exact-copy.jpg"),
                class_index=duplicate.class_index,
                exact_hash=duplicate.exact_hash,
                perceptual_hash=duplicate.perceptual_hash,
            )
        )

        train_indices, validation_indices = grouped_validation_indices(samples, 0.25, 7)

        self.assertTrue(train_indices)
        self.assertTrue(validation_indices)
        self.assertEqual(0 in train_indices, len(samples) - 1 in train_indices)
        self.assertEqual(0 in validation_indices, len(samples) - 1 in validation_indices)
        self.assertEqual(
            {samples[index].class_index for index in validation_indices},
            set(range(len(FACE_CLASS_NAMES))),
        )

    def test_overlap_audit_flags_exact_cross_split_duplicates(self) -> None:
        train_sample = ImageSample(Path("train.jpg"), 0, "same", 123)
        test_sample = ImageSample(Path("test.jpg"), 0, "same", 123)

        audit = audit_train_test_overlap([train_sample], [test_sample])

        self.assertEqual(audit["exact_duplicate_pairs"], 1)
        self.assertIsNotNone(audit["warning"])

    def test_training_duplicate_audit_reports_conflicting_labels(self) -> None:
        duplicate_group = [
            ImageSample(Path("anger.jpg"), 0, "same-pixels", 1),
            ImageSample(Path("sad.jpg"), 4, "same-pixels", 1),
        ]

        audit = audit_training_duplicates(duplicate_group)

        self.assertEqual(audit["exact_duplicate_groups"], 1)
        self.assertEqual(audit["extra_exact_duplicate_files"], 1)
        self.assertEqual(audit["cross_label_exact_duplicate_groups"], 1)

    def test_metrics_return_per_class_scores_and_confusion_matrix(self) -> None:
        result = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])

        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"][0][:2], [1, 1])
        self.assertEqual(result["confusion_matrix"][1][:2], [0, 2])
        self.assertEqual(result["per_class"][FACE_CLASS_NAMES[0]]["support"], 2)
        self.assertAlmostEqual(result["per_class"][FACE_CLASS_NAMES[0]]["precision"], 1.0)
        self.assertAlmostEqual(result["per_class"][FACE_CLASS_NAMES[0]]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
