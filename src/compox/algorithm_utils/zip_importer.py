"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from compox.exceptions import CompoxImportError, CompoxValidationError


class ZipImporter:
    """
    Import a Runner module from zip archive bytes while keeping files on disk.

    The archive is persisted and extracted under a shared cache directory so code
    remains importable for subprocesses that need to re-import algorithm modules.

    Parameters
    ----------
    zip_bytes : bytes
        Bytes of the module zip archive.
    module_id : str
        Stable module identifier used for cache names.
    """

    _lock = threading.RLock()
    _root_cache_dir = Path(tempfile.gettempdir()) / "compox" / "module_cache"
    _instance_id = f"pid_{os.getpid()}_{uuid4().hex[:8]}"
    _cache_dir = _root_cache_dir / _instance_id

    def __init__(self, zip_bytes: bytes, module_id: str):
        self.zip_bytes = zip_bytes
        self.module_id = module_id
        self._module_root: Path | None = None
        self._module: ModuleType | None = None

    @classmethod
    def configure_cache_dir(
        cls,
        cache_dir: str | os.PathLike,
        instance_id: str | None = None,
    ) -> None:
        """
        Set root cache directory and initialize an instance-specific cache path.

        Parameters
        ----------
        cache_dir : str | os.PathLike
            Root directory path where instance cache folders are created.
        instance_id : str | None, optional
            Optional stable identifier for current instance cache folder.
            If not provided, generated automatically.
        """
        with cls._lock:
            cls._root_cache_dir = Path(cache_dir)
            if instance_id is not None:
                cls._instance_id = cls._safe_module_name(instance_id)
            cls._root_cache_dir.mkdir(parents=True, exist_ok=True)
            cls.cleanup_root_cache()
            cls._cache_dir = cls._root_cache_dir / cls._instance_id
            cls._cache_dir.mkdir(parents=True, exist_ok=True)
            cls._harden_runtime_permissions(cls._root_cache_dir)
            cls._harden_runtime_permissions(cls._cache_dir)

    @classmethod
    def cleanup_cache(cls) -> None:
        """
        Remove cached module entries for the current runtime instance.
        """
        with cls._lock:
            if not cls._cache_dir.exists():
                return

            try:
                shutil.rmtree(cls._cache_dir, ignore_errors=True)
            except FileNotFoundError:
                return

    @classmethod
    def cleanup_root_cache(cls) -> None:
        """
        Remove all runtime cache entries under the configured root cache path.
        """
        with cls._lock:
            if not cls._root_cache_dir.exists():
                return

            for entry in cls._root_cache_dir.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                except FileNotFoundError:
                    continue

    def __enter__(self) -> ModuleType:
        """
        Extract and import the module Runner file as a Python module.

        Returns
        -------
        ModuleType
            Imported module object containing `Runner`.
        """
        self._module_root = self._ensure_extracted_module()
        self._verify_module_integrity(self._module_root)
        runner_file = self._find_runner_file(self._module_root)

        if str(self._module_root) not in sys.path:
            # Keep path in sys.path for potential subprocess re-imports.
            sys.path.insert(0, str(self._module_root))

        package_module_name = self._build_package_module_name(
            self._module_root, runner_file
        )
        if package_module_name is not None:
            module = importlib.import_module(package_module_name)
        else:
            module_name = self._build_unique_module_name(self.module_id)
            spec = importlib.util.spec_from_file_location(
                module_name, runner_file
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    "Failed to build import spec for runner in module "
                    f"'{self.module_id}'."
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        self._module = module
        return module

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context manager exit handler.

        Returns
        -------
        bool
            Always False to propagate any exception from import/use.
        """
        return False

    @classmethod
    def _safe_module_name(cls, module_id: str) -> str:
        """Convert arbitrary module id into a valid filesystem-friendly name."""
        cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", module_id)
        if not cleaned or cleaned[0].isdigit():
            cleaned = f"m_{cleaned}"
        return cleaned

    @classmethod
    def _build_unique_module_name(cls, module_id: str) -> str:
        """Build a unique Python module name used in `sys.modules`."""
        return f"compox_runner_{cls._safe_module_name(module_id)}"

    @classmethod
    def _build_package_module_name(
        cls, module_root: Path, runner_file: Path
    ) -> str | None:
        """
        Build dotted module path when Runner belongs to a package tree.

        Returns
        -------
        str | None
            Dotted package module path (e.g. `pkg.Runner`) or None when
            package structure is not detected.
        """
        try:
            rel_parts = list(runner_file.relative_to(module_root).parts)
        except ValueError:
            return None

        if len(rel_parts) < 2:
            return None
        if rel_parts[-1] not in ("Runner.py", "Runner.pyc"):
            return None

        package_parts = rel_parts[:-1]
        package_dir = module_root.joinpath(*package_parts)
        if not (
            package_dir.joinpath("__init__.py").exists()
            or package_dir.joinpath("__init__.pyc").exists()
        ):
            return None

        for depth in range(1, len(package_parts)):
            parent = module_root.joinpath(*package_parts[:depth])
            if not (
                parent.joinpath("__init__.py").exists()
                or parent.joinpath("__init__.pyc").exists()
            ):
                return None

        return ".".join([*package_parts, "Runner"])

    @classmethod
    def _cache_entry_dir(cls, module_id: str) -> Path:
        """Return cache directory path for a given module identifier."""
        return cls._cache_dir / cls._safe_module_name(module_id)

    @classmethod
    def _is_safe_zip_member(cls, member_name: str) -> bool:
        """Validate zip member path to prevent path traversal extraction."""
        member_path = Path(member_name)
        if member_path.is_absolute():
            return False
        return ".." not in member_path.parts

    @classmethod
    def _extract_zip_bytes(cls, zip_bytes: bytes, target_dir: Path) -> None:
        """
        Extract zip bytes into target directory after path safety checks.

        Raises
        ------
        ValueError
            If archive contains unsafe member path.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            for member in archive.namelist():
                if not cls._is_safe_zip_member(member):
                    raise CompoxValidationError(
                        f"Unsafe zip member path: {member}",
                        code="unsafe_zip_member_path",
                        details={"member": member},
                    )
            archive.extractall(target_dir)
        cls._harden_tree_permissions(target_dir)

    @classmethod
    def _compute_zip_expected_member_hashes(
        cls, zip_bytes: bytes
    ) -> dict[str, str]:
        """
        Compute expected file content hashes from zip bytes.

        Returns a mapping of relative POSIX path -> SHA256(content) for all
        non-directory archive members with safe paths.
        """
        expected: dict[str, str] = {}
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            members = sorted(
                m
                for m in archive.namelist()
                if cls._is_safe_zip_member(m) and not m.endswith("/")
            )
            for member in members:
                content = archive.read(member)
                expected[member] = hashlib.sha256(content).hexdigest()
        return expected

    @classmethod
    def _compute_extracted_member_hashes(
        cls, extract_dir: Path, rel_paths: list[str]
    ) -> dict[str, str]:
        """
        Compute hashes for selected extracted files.

        Only paths listed in `rel_paths` are validated; extra runtime artifacts
        (e.g. __pycache__) are intentionally ignored.
        """
        actual: dict[str, str] = {}
        for rel_path in sorted(rel_paths):
            fs_path = extract_dir / Path(rel_path)
            if not fs_path.exists() or not fs_path.is_file():
                raise CompoxImportError(
                    "Module integrity verification failed: expected file "
                    f"'{rel_path}' missing in '{extract_dir}'.",
                    code="module_cache_integrity_missing_file",
                    details={
                        "relative_path": rel_path,
                        "extract_dir": str(extract_dir),
                    },
                )
            actual[rel_path] = hashlib.sha256(fs_path.read_bytes()).hexdigest()
        return actual

    def _verify_module_integrity(self, module_root: Path) -> None:
        """
        Verify extracted files match expected archive member contents.
        """
        expected = self._compute_zip_expected_member_hashes(self.zip_bytes)
        actual = self._compute_extracted_member_hashes(
            module_root, list(expected.keys())
        )
        if actual != expected:
            raise CompoxImportError(
                "Module integrity verification failed for "
                f"module_id='{self.module_id}'.",
                code="module_cache_integrity_mismatch",
                details={"module_id": self.module_id},
            )

    @classmethod
    def _find_runner_file(cls, module_root: Path) -> Path:
        """
        Locate ``Runner.py`` or ``Runner.pyc`` in extracted module tree.

        Raises
        ------
        FileNotFoundError
            If no supported Runner module file is present.
        """
        direct = module_root / "Runner.py"
        if direct.exists():
            return direct
        direct_pyc = module_root / "Runner.pyc"
        if direct_pyc.exists():
            return direct_pyc

        matches = list(module_root.rglob("Runner.py"))
        if not matches:
            matches = list(module_root.rglob("Runner.pyc"))
        if not matches:
            raise FileNotFoundError(
                f"Runner.py / Runner.pyc not found in extracted module at '{module_root}'."
            )
        return matches[0]

    def _ensure_extracted_module(self) -> Path:
        """
        Ensure module archive is cached and extracted for current runtime.

        Returns
        -------
        Path
            Directory containing extracted module files.
        """
        with self._lock:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            entry_dir = self._cache_entry_dir(self.module_id)
            extract_dir = entry_dir / "module"

            if not entry_dir.exists():
                entry_dir.mkdir(parents=True, exist_ok=True)
                self._harden_runtime_permissions(entry_dir)

            if not extract_dir.exists():
                self._extract_zip_bytes(self.zip_bytes, extract_dir)
            self._harden_runtime_permissions(entry_dir)
            self._harden_runtime_permissions(extract_dir)
            return extract_dir

    @classmethod
    def _harden_runtime_permissions(cls, path: Path) -> None:
        """
        Best-effort tightening of runtime path permissions.

        On POSIX: enforce owner-only read/write/execute for directories
        and owner-only read/write for files. On non-POSIX systems this is
        a best-effort operation and may provide limited guarantees.
        """
        try:
            if os.name == "posix":
                if path.is_dir():
                    path.chmod(0o700)
                elif path.exists():
                    path.chmod(0o600)
            elif path.exists() and path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)
        except Exception:
            # Permission hardening is best-effort and should not break runtime.
            return

    @classmethod
    def _harden_tree_permissions(cls, root: Path) -> None:
        """
        Best-effort recursive permission hardening for extracted module tree.
        """
        cls._harden_runtime_permissions(root)
        if not root.exists():
            return
        for item in root.rglob("*"):
            cls._harden_runtime_permissions(item)
