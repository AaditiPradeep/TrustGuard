"""Backward-compatible import shim for the canonical TrustGuard schema.

The project now has one source of truth for evidence records: ``data.schema``.
This module is retained only so any older teammate code importing
``data.evidence_schema`` keeps working without creating a second, incompatible
EvidencePassage definition.
"""

from data.schema import EvidencePassage, write_jsonl


def save_jsonl(records, output_file):
    """Compatibility alias for older teammate code."""
    write_jsonl(records, output_file)


__all__ = ["EvidencePassage", "save_jsonl"]
