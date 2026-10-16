"""aucml: a shared ML platform for many models.

Domain knowledge lives in ``feature_repo/`` (features) and ``models/`` (one YAML
spec per model). Nothing in this package may name a model or a feature group;
``tests/test_platform_hygiene.py`` enforces that.
"""

__version__ = "0.1.0"
