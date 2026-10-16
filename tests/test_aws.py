"""AWS adapters against botocore's Stubber: request shapes in, response shapes out."""

from __future__ import annotations

import boto3
import pandas as pd
from botocore.stub import Stubber

from aucml.aws import SageMakerOnlineStore, group_name, register_model_package


def _client(service: str):  # type: ignore[no-untyped-def]
    return boto3.client(service, region_name="eu-central-1", aws_access_key_id="x", aws_secret_access_key="x")


def test_online_lookup_maps_batch_get_record() -> None:
    client = _client("sagemaker-featurestore-runtime")
    store = SageMakerOnlineStore("aucml-prod", client)
    with Stubber(client) as stub:
        stub.add_response(
            "batch_get_record",
            {
                "Records": [
                    {
                        "FeatureGroupName": "aucml-prod-seller-90d",
                        "RecordIdentifierValueAsString": "S0001",
                        "Record": [
                            {"FeatureName": "entity_key", "ValueAsString": "S0001"},
                            {"FeatureName": "event_time", "ValueAsString": "2025-12-01T00:00:00"},
                            {"FeatureName": "known_at", "ValueAsString": "2025-11-30T17:00:00"},
                            {"FeatureName": "sell_through_90d", "ValueAsString": "0.71"},
                        ],
                    }
                ],
                "Errors": [],
                "UnprocessedIdentifiers": [],
            },
            {"Identifiers": [{"FeatureGroupName": "aucml-prod-seller-90d", "RecordIdentifiersValueAsString": ["S0001", "S0002"]}]},
        )
        got = store.get("seller_90d", pd.Series(["S0001", "S0002", "S0001"]), pd.Series([pd.Timestamp("2025-12-02")] * 3))
    assert got.sell_through_90d.iloc[0] == 0.71 and pd.isna(got.sell_through_90d.iloc[1])
    assert got.event_time.iloc[2] == pd.Timestamp("2025-12-01")


def test_put_record_skips_missing_values() -> None:
    client = _client("sagemaker-featurestore-runtime")
    store = SageMakerOnlineStore("aucml-prod", client)
    with Stubber(client) as stub:
        stub.add_response(
            "put_record",
            {},
            {
                "FeatureGroupName": "aucml-prod-lane-cost-56d",
                "Record": [
                    {"FeatureName": "entity_key", "ValueAsString": "DE>FR"},
                    {"FeatureName": "event_time", "ValueAsString": "2025-12-01T00:00:00"},
                    {"FeatureName": "median_cost_56d", "ValueAsString": "512.5"},
                ],
            },
        )
        n = store.put(
            "lane_cost_56d",
            pd.DataFrame(
                {"entity_key": ["DE>FR"], "event_time": [pd.Timestamp("2025-12-01")], "median_cost_56d": [512.5], "volume_56d": [float("nan")]}
            ),
        )
    assert n == 1


def test_model_package_carries_gate_verdict() -> None:
    client = _client("sagemaker")
    manifest = {
        "name": "price_estimator",
        "version": "price_estimator-20251101-abc",
        "training_window": {"cutoff": "2025-11-01"},
        "primary_metric": "mape",
        "metrics": {"mape": 0.0795},
    }
    with Stubber(client) as stub:
        stub.add_response(
            "create_model_package",
            {"ModelPackageArn": "arn:aws:sagemaker:eu-central-1:123456789012:model-package/aucml-prod-price-estimator/1"},
            {
                "ModelPackageGroupName": "aucml-prod-price-estimator",
                "ModelPackageDescription": "price_estimator-20251101-abc trained to 2025-11-01",
                "InferenceSpecification": {
                    "Containers": [{"Image": "img:1", "ModelDataUrl": "s3://b/model.tar.gz"}],
                    "SupportedContentTypes": ["application/json"],
                    "SupportedResponseMIMETypes": ["application/json"],
                },
                "ModelApprovalStatus": "Approved",
                "CustomerMetadataProperties": {"version": "price_estimator-20251101-abc", "primary_metric": "mape", "primary_value": "0.0795"},
            },
        )
        arn = register_model_package(manifest, "s3://b/model.tar.gz", "img:1", "aucml-prod", approved=True, client=client)
    assert arn.endswith("/1")


def test_group_names_match_terraform() -> None:
    assert group_name("aucml-prod", "seller_90d") == "aucml-prod-seller-90d"
