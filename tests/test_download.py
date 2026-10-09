import stat
from hashlib import sha256
from pathlib import Path

import pytest

from dice_detector.download import Artifact, LicenseNotAcceptedError, download_artifacts


class Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_download_requires_explicit_license_acceptance(tmp_path: Path) -> None:
    with pytest.raises(LicenseNotAcceptedError):
        download_artifacts(tmp_path, accept_license=False)


def test_download_verifies_checksum_and_reuses_valid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model"
    artifact = Artifact("model.onnx", "https://example.test/model", sha256(payload).hexdigest())
    calls = 0

    def urlopen(*args: object, **kwargs: object) -> Response:
        nonlocal calls
        calls += 1
        return Response(payload)

    monkeypatch.setattr("dice_detector.download.urlopen", urlopen)

    paths = download_artifacts(
        tmp_path,
        accept_license=True,
        artifacts=(artifact,),
    )
    second_paths = download_artifacts(
        tmp_path,
        accept_license=True,
        artifacts=(artifact,),
    )

    assert paths == second_paths == (tmp_path / "model.onnx",)
    assert (tmp_path / "model.onnx").read_bytes() == payload
    assert stat.S_IMODE((tmp_path / "model.onnx").stat().st_mode) == 0o644
    assert calls == 1


def test_download_removes_partial_file_on_checksum_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = Artifact("model.onnx", "https://example.test/model", "0" * 64)
    monkeypatch.setattr("dice_detector.download.urlopen", lambda *args, **kwargs: Response(b"bad"))

    with pytest.raises(ValueError, match="checksum"):
        download_artifacts(tmp_path, accept_license=True, artifacts=(artifact,))

    assert list(tmp_path.iterdir()) == []


def test_download_rejects_non_https_url(tmp_path: Path) -> None:
    artifact = Artifact("model.onnx", "file:///tmp/model", "0" * 64)

    with pytest.raises(ValueError, match="HTTPS"):
        download_artifacts(tmp_path, accept_license=True, artifacts=(artifact,))
