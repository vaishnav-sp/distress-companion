"""Phase 3 multimodal evidence aggregation package."""

from .fusion import FusionEvidence, build_fusion_evidence
from .schema import (
    FEATURE_SCHEMA_VERSION,
    FeatureRecord,
    FeatureSchema,
    build_fusion_diagnostic_snapshot,
    evidence_to_feature_record,
    feature_order,
)

__all__ = [
    "FusionEvidence",
    "FeatureRecord",
    "FeatureSchema",
    "FEATURE_SCHEMA_VERSION",
    "build_fusion_evidence",
    "build_fusion_diagnostic_snapshot",
    "evidence_to_feature_record",
    "feature_order",
]
