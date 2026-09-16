"""
Copyright 2026 Tescan GROUP, a.s.
All rights reserved
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from zipfile import ZIP_DEFLATED, ZipFile

from compox.database_connection.BaseConnection import BaseConnection
from compox.exceptions import CompoxBundleError, CompoxConfigurationError

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as e:  # pragma: no cover
    AESGCM = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class CompoxAlgorithmBundleBuilder:
    """
    Build an encrypted snapshot of builtin algorithm-related storage collections.

    The generated bundle is intended for distribution and later import into a
    Compox backend. By default it snapshots:
    - `algorithm-store`
    - `module-store`
    - `asset-store`
    - `algorithm-checkpoint-store`
    """

    def __init__(self, source_db: BaseConnection, bundle_key: str):
        """
        Initialize encrypted bundle builder.

        Parameters
        ----------
        source_db : BaseConnection
            Source storage connection used for export.
        bundle_key : str
            Base64url-encoded 32-byte AES key.
        """
        if AESGCM is None:
            raise CompoxConfigurationError(
                "cryptography library is required for CompoxAlgorithmBundleBuilder",
                code="bundle_crypto_dependency_missing",
                cause=_IMPORT_ERROR,
            ) from _IMPORT_ERROR
        self._source_db = source_db
        self._key = self._decode_key(bundle_key)
        self._aesgcm = AESGCM(self._key)

    def build(
        self,
        output_zip_path: str,
        collections: list[str] | None = None,
    ) -> int:
        """
        Build an encrypted algorithm bundle zip from selected collections.

        Parameters
        ----------
        output_zip_path : str
            Output path for the algorithm bundle zip.
        collections : list[str] | None, optional
            Storage collections to export. By default the builtin algorithm
            snapshot collections are used.

        Returns
        -------
        int
            Number of bundled objects.
        """
        if collections is None:
            collections = [
                "algorithm-store",
                "module-store",
                "asset-store",
                "algorithm-checkpoint-store",
            ]
        existing_collections = set(self._source_db.list_collections())
        selected_collections = [
            collection
            for collection in collections
            if collection in existing_collections
        ]

        objects_manifest: list[dict] = []
        os.makedirs(
            os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True
        )

        with ZipFile(output_zip_path, "w", compression=ZIP_DEFLATED) as zf:
            obj_index = 0
            for collection in selected_collections:
                keys = self._normalize_keys(
                    self._source_db.list_objects(collection)
                )
                if not keys:
                    continue
                payloads = self._source_db.get_objects(collection, keys)
                for key, payload in zip(keys, payloads):
                    if isinstance(payload, str):
                        payload = payload.encode("utf-8")
                    nonce = os.urandom(12)
                    ciphertext = self._aesgcm.encrypt(nonce, payload, None)
                    blob_path = f"objects/{obj_index:08d}.bin"
                    obj_index += 1
                    zf.writestr(blob_path, ciphertext)
                    objects_manifest.append(
                        {
                            "collection": collection,
                            "key": key,
                            "blob": blob_path,
                            "nonce_b64": base64.b64encode(nonce).decode(
                                "ascii"
                            ),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size": len(payload),
                        }
                    )

            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "compox_version": self._get_compox_version(),
                "bundle_kind": "builtin_algorithm_snapshot",
            }
            content_sha256 = self._compute_content_sha256(
                selected_collections, objects_manifest, metadata
            )
            manifest = {
                "format": "compox-migration-bundle-v1",
                "encryption": "aesgcm",
                "collections": sorted(set(selected_collections)),
                "metadata": {
                    **metadata,
                    "content_sha256": content_sha256,
                },
                "objects": objects_manifest,
            }
            manifest_bytes = json.dumps(
                manifest, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            manifest_hmac_hex = hmac.new(
                self._key, manifest_bytes, hashlib.sha256
            ).hexdigest()
            zf.writestr("manifest.json", manifest_bytes)
            zf.writestr("manifest.hmac", manifest_hmac_hex.encode("utf-8"))

        return len(objects_manifest)

    @staticmethod
    def generate_key() -> str:
        """
        Generate a base64url-encoded AES-256 bundle key.
        """
        return (
            base64.urlsafe_b64encode(os.urandom(32)).decode("ascii").rstrip("=")
        )

    @staticmethod
    def _normalize_keys(items: list[dict] | list[str]) -> list[str]:
        """
        Normalize `list_objects()` output to plain key strings.
        """
        keys = []
        for item in items:
            if isinstance(item, dict) and "Key" in item:
                keys.append(str(item["Key"]))
            else:
                keys.append(str(item))
        return keys

    @staticmethod
    def _decode_key(bundle_key: str) -> bytes:
        """
        Decode a base64url bundle key into raw AES-256 key bytes.
        """
        key = bundle_key.strip()
        try:
            raw = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
        except (ValueError, TypeError) as e:
            raise CompoxBundleError(
                "Invalid bundle key format. Expected base64url-encoded 32-byte key.",
                code="invalid_bundle_key",
                http_status=400,
                cause=e,
            ) from e
        if len(raw) != 32:
            raise CompoxBundleError(
                f"Invalid bundle key length: expected 32 bytes, got {len(raw)} bytes.",
                code="invalid_bundle_key",
                http_status=400,
                details={"expected_bytes": 32, "actual_bytes": len(raw)},
            )
        return raw

    @staticmethod
    def _compute_content_sha256(
        collections: list[str],
        objects_manifest: list[dict],
        metadata: dict[str, str],
    ) -> str:
        """
        Compute a deterministic fingerprint of the bundle manifest content.
        """
        content_descriptor = {
            "format": "compox-migration-bundle-v1",
            "encryption": "aesgcm",
            "collections": sorted(set(collections)),
            "metadata": metadata,
            "objects": objects_manifest,
        }
        payload = json.dumps(
            content_descriptor, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _get_compox_version() -> str:
        """
        Read the installed Compox package version.
        """
        try:
            return importlib.metadata.version("compox")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
