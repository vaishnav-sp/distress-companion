"""MobileNetV3-Small construction and preprocessing shared by train and inference."""

from typing import Any

from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import InterpolationMode

from config import FACE_INPUT_SIZE

ARCHITECTURE = "mobilenet_v3_small"
WEIGHT_ENUM = MobileNet_V3_Small_Weights.IMAGENET1K_V1
WEIGHT_SOURCE = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def build_face_model(num_classes: int, pretrained: bool):
    """Create MobileNetV3-Small and replace its final classifier layer."""
    weights = WEIGHT_ENUM if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    final_layer = model.classifier[-1]
    model.classifier[-1] = type(final_layer)(final_layer.in_features, num_classes)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    return model


def make_training_transform() -> transforms.Compose:
    """Use augmentation only on training data, then ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                FACE_INPUT_SIZE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )


def make_evaluation_transform() -> Any:
    """Match the preprocessing documented for the selected pretrained weights."""
    return WEIGHT_ENUM.transforms()


def preprocessing_metadata() -> dict[str, Any]:
    """Return a JSON-serializable description of inference preprocessing."""
    return {
        "input_color": "RGB",
        "resize_shorter_side": 256,
        "center_crop": [FACE_INPUT_SIZE, FACE_INPUT_SIZE],
        "interpolation": "bilinear",
        "scale": "uint8 [0, 255] to float [0.0, 1.0]",
        "mean": list(NORMALIZATION_MEAN),
        "std": list(NORMALIZATION_STD),
        "training_augmentation": [
            "RandomResizedCrop(224, bilinear)",
            "RandomHorizontalFlip(p=0.5)",
        ],
        "source": "Torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()",
    }
