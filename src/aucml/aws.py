"""AWS adapters: SageMaker Feature Store (online reads/writes) and Model Registry.

Same interfaces as the local stand-ins, so the serving path and the release
command do not change between laptop, CI and AWS. Tested against botocore's
Stubber (request/response shapes), not against a live account.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from aucml.featurestore import _jsonable


def _client(service: str) -> Any:
    import boto3

    return boto3.client(service)


def group_name(prefix: str, group: str) -> str:
    """Feature group naming used by infra/terraform/modules/feature_group."""
    return f"{prefix}-{group.replace('_', '-')}"[:64]


class SageMakerOnlineStore:
    """Online lookups via sagemaker-featurestore-runtime BatchGetRecord."""

    MAX_IDS = 100  # BatchGetRecord limit per request

    def __init__(self, prefix: str, client: Any | None = None) -> None:
        self.prefix = prefix
        self.client = client or _client("sagemaker-featurestore-runtime")

    @classmethod
    def from_env(cls) -> SageMakerOnlineStore:
        return cls(os.environ["AUCML_FG_PREFIX"])

    def get(self, group: str, keys: pd.Series, as_of: pd.Series) -> pd.DataFrame:
        del as_of
        name = group_name(self.prefix, group)
        uniq = [str(k) for k in pd.unique(keys.astype(str))]
        found: dict[str, dict[str, object]] = {}
        for i in range(0, len(uniq), self.MAX_IDS):
            resp = self.client.batch_get_record(
                Identifiers=[{"FeatureGroupName": name, "RecordIdentifiersValueAsString": uniq[i : i + self.MAX_IDS]}]
            )
            for rec in resp.get("Records", []):
                found[rec["RecordIdentifierValueAsString"]] = {f["FeatureName"]: f["ValueAsString"] for f in rec["Record"]}
            if resp.get("Errors"):
                raise RuntimeError(f"feature store errors: {resp['Errors'][:3]}")
        rows = [found.get(str(k), {}) for k in keys]
        df = pd.DataFrame.from_records(rows, index=keys.index)
        for c in df.columns:
            if c not in ("entity_key", "event_time", "known_at"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ("event_time", "known_at"):
            df[c] = pd.to_datetime(df[c], format="ISO8601") if c in df else pd.NaT
        return df.drop(columns=["entity_key"], errors="ignore")

    def put(self, group: str, snapshot: pd.DataFrame) -> int:
        """Write the latest snapshot rows (entity_key, event_time, known_at, features...)."""
        name = group_name(self.prefix, group)
        n = 0
        for row in snapshot.to_dict(orient="records"):
            record = [{"FeatureName": k, "ValueAsString": str(_jsonable(v))} for k, v in row.items() if _jsonable(v) is not None]
            self.client.put_record(FeatureGroupName=name, Record=record)
            n += 1
        return n


def register_model_package(
    manifest: dict[str, Any],
    model_data_url: str,
    image_uri: str,
    group_prefix: str,
    approved: bool,
    client: Any | None = None,
) -> str:
    """Register a packaged version in the SageMaker Model Registry.

    The gate decision maps to ModelApprovalStatus, so the registry in AWS shows
    the same verdict as ``registry/<model>/<version>/gate.json``.
    """
    sm = client or _client("sagemaker")
    resp = sm.create_model_package(
        ModelPackageGroupName=f"{group_prefix}-{manifest['name'].replace('_', '-')}",
        ModelPackageDescription=f"{manifest['version']} trained to {manifest['training_window']['cutoff']}",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved" if approved else "Rejected",
        CustomerMetadataProperties={
            "version": manifest["version"],
            "primary_metric": manifest["primary_metric"],
            "primary_value": json.dumps(manifest["metrics"][manifest["primary_metric"]]),
        },
    )
    return str(resp["ModelPackageArn"])
