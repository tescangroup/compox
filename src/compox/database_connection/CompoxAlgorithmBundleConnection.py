"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from zipfile import BadZipFile, ZipFile

from compox.database_connection.BaseConnection import BaseConnection
from compox.exceptions import CompoxBundleError, CompoxConfigurationError

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as e:  # pragma: no cover
    AESGCM = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class CompoxAlgorithmBundleConnection(BaseConnection):
    """
    Read-only connection for encrypted Compox algorithm bundles.

    Bundle format (zip):
      - manifest.json
      - manifest.hmac (hex HMAC-SHA256 of manifest.json bytes)
      - encrypted blobs referenced by manifest objects[*].blob

    Manifest schema (v1):
      {
        "format": "compox-migration-bundle-v1",
        "encryption": "aesgcm",
        "collections": [
          "algorithm-store",
          "module-store",
          "asset-store",
          "algorithm-checkpoint-store"
        ],
        "metadata": {
          "created_at": "...",
          "compox_version": "...",
          "bundle_kind": "builtin_algorithm_snapshot",
          "content_sha256": "..."
        },
        "objects": [
          {
            "collection": "algorithm-store",
            "key": "algo-id~name~1",
            "blob": "objects/0001.bin",
            "nonce_b64": "...",
            "sha256": "...",
            "size": 1234
          }
        ]
      }
    """

    def __init__(self, bundle_path: str, bundle_key: str):
        """
        Initialize a read-only encrypted bundle reader.

        Parameters
        ----------
        bundle_path : str
            Path to algorithm bundle zip file.
        bundle_key : str
            Base64url-encoded 32-byte AES key.
        """
        super().__init__()
        if AESGCM is None:
            raise CompoxConfigurationError(
                "cryptography library is required for CompoxAlgorithmBundleConnection",
                code="bundle_crypto_dependency_missing",
                cause=_IMPORT_ERROR,
            ) from _IMPORT_ERROR

        self._bundle_path = bundle_path
        self._key = self._decode_key(bundle_key)
        self._aesgcm = AESGCM(self._key)
        self._manifest = self._load_manifest()
        self._objects_index = self._build_objects_index()

    def list_collections(self) -> list[str]:
        """
        List collection names declared by the bundle manifest.
        """
        declared = self._manifest.get("collections")
        if isinstance(declared, list):
            return sorted(str(name) for name in declared)
        return sorted(
            {entry["collection"] for entry in self._manifest["objects"]}
        )

    def check_collections_exists(
        self, collection_names: list[str]
    ) -> list[bool]:
        """
        Check collection existence against the bundle manifest.
        """
        existing = set(self.list_collections())
        return [name in existing for name in collection_names]

    def delete_collections(self, collection_names: list[str]) -> None:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def create_collections(self, collection_names: list[str]) -> None:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def list_objects(self, collection_name: str) -> list[str]:
        """
        List object keys in the selected bundle collection.
        """
        return sorted(list(self._objects_index.get(collection_name, {}).keys()))

    def check_objects_exist(
        self, collection_name: str, object_names: list[str]
    ) -> list[bool]:
        """
        Check whether object keys exist in the selected bundle collection.
        """
        collection = self._objects_index.get(collection_name, {})
        return [name in collection for name in object_names]

    def get_objects(
        self, collection_name: str, object_names: list[str]
    ) -> list[bytes]:
        """
        Read, decrypt and validate objects from the bundle.
        """
        collection = self._objects_index.get(collection_name, {})
        outputs = []
        with ZipFile(self._bundle_path, "r") as zf:
            for name in object_names:
                if name not in collection:
                    raise CompoxBundleError(
                        f"Object '{name}' not found in collection '{collection_name}'",
                        code="bundle_object_not_found",
                        http_status=404,
                        details={
                            "collection": collection_name,
                            "object": name,
                        },
                    )
                entry = collection[name]
                try:
                    ciphertext = zf.read(entry["blob"])
                    nonce = base64.b64decode(entry["nonce_b64"])
                    plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
                except Exception as e:
                    raise CompoxBundleError(
                        f"Failed to read encrypted bundle object '{collection_name}/{name}'",
                        code="bundle_object_read_failed",
                        details={
                            "collection": collection_name,
                            "object": name,
                        },
                        cause=e,
                    ) from e
                digest = hashlib.sha256(plaintext).hexdigest()
                if digest != entry["sha256"]:
                    raise CompoxBundleError(
                        f"Checksum mismatch for '{collection_name}/{name}'",
                        code="bundle_checksum_mismatch",
                        details={
                            "collection": collection_name,
                            "object": name,
                        },
                    )
                outputs.append(plaintext)
        return outputs

    def put_objects(
        self,
        collection_name: str,
        object_names: list[str],
        object: list[bytes] | list[str],
    ) -> None:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def put_objects_with_duplicity_check(
        self, collection_name: str, object_names: list[str], object: list[bytes]
    ) -> list[bool]:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def delete_objects(
        self, collection_name: str, object_names: list[str]
    ) -> None:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def get_object_tags(
        self, collection_name: str, object_name: str
    ) -> dict[str, str]:
        return {}

    def put_object_tags(
        self, collection_name: str, object_name: str, tags: dict[str, str]
    ) -> None:
        raise NotImplementedError(
            "CompoxAlgorithmBundleConnection is read-only."
        )

    def get_bundle_info(self) -> dict:
        """
        Return lightweight bundle metadata for CLI inspection.
        """
        object_counts: dict[str, int] = {}
        for entry in self._manifest["objects"]:
            collection = str(entry["collection"])
            object_counts[collection] = object_counts.get(collection, 0) + 1

        return {
            "format": self._manifest.get("format"),
            "encryption": self._manifest.get("encryption"),
            "collections": self.list_collections(),
            "metadata": self._manifest.get("metadata", {}),
            "object_count": len(self._manifest["objects"]),
            "object_counts_by_collection": object_counts,
        }

    def _load_manifest(self) -> dict:
        """
        Load and validate the bundle manifest and its HMAC signature.
        """
        try:
            with ZipFile(self._bundle_path, "r") as zf:
                manifest_bytes = zf.read("manifest.json")
                manifest_hmac_hex = (
                    zf.read("manifest.hmac").decode("utf-8").strip()
                )
        except (BadZipFile, KeyError, UnicodeDecodeError) as e:
            raise CompoxBundleError(
                "Invalid Compox algorithm bundle archive.",
                code="invalid_algorithm_bundle",
                http_status=400,
                details={"path": self._bundle_path},
                cause=e,
            ) from e
        calc = hmac.new(self._key, manifest_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calc, manifest_hmac_hex):
            raise CompoxBundleError(
                "Invalid bundle HMAC signature for manifest.json",
                code="invalid_bundle_signature",
                http_status=400,
            )

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise CompoxBundleError(
                "Invalid bundle manifest JSON.",
                code="invalid_bundle_manifest",
                http_status=400,
                cause=e,
            ) from e
        if manifest.get("format") != "compox-migration-bundle-v1":
            raise CompoxBundleError(
                "Unsupported bundle format",
                code="unsupported_bundle_format",
                http_status=400,
            )
        if manifest.get("encryption") != "aesgcm":
            raise CompoxBundleError(
                "Unsupported bundle encryption",
                code="unsupported_bundle_encryption",
                http_status=400,
            )
        collections = manifest.get("collections")
        if collections is not None and not isinstance(collections, list):
            raise CompoxBundleError(
                "Invalid bundle manifest: collections must be a list",
                code="invalid_bundle_manifest",
                http_status=400,
            )
        metadata = manifest.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise CompoxBundleError(
                "Invalid bundle manifest: metadata must be a dict",
                code="invalid_bundle_manifest",
                http_status=400,
            )
        if not isinstance(manifest.get("objects"), list):
            raise CompoxBundleError(
                "Invalid bundle manifest: missing objects list",
                code="invalid_bundle_manifest",
                http_status=400,
            )
        return manifest

    def _build_objects_index(self) -> dict[str, dict[str, dict]]:
        """
        Build an in-memory index: collection -> object key -> manifest entry.
        """
        index: dict[str, dict[str, dict]] = {}
        for entry in self._manifest["objects"]:
            collection = str(entry["collection"])
            key = str(entry["key"])
            index.setdefault(collection, {})
            index[collection][key] = entry
        return index

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
                "Invalid bundle key encoding",
                code="invalid_bundle_key",
                http_status=400,
                cause=e,
            ) from e
        if len(raw) != 32:
            raise CompoxBundleError(
                "Bundle key must decode to 32 bytes (AES-256 key)",
                code="invalid_bundle_key",
                http_status=400,
                details={"expected_bytes": 32, "actual_bytes": len(raw)},
            )
        return raw
