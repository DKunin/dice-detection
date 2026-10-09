import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MODEL_REVISION = "35480b868964b3044668d0d8fbb86b5e063ca409"
MODEL_CARD_URL = "https://huggingface.co/G-G-Games/diecamera-models"
MODEL_BASE_URL = "https://huggingface.co/G-G-Games/diecamera-models/resolve/" + MODEL_REVISION


@dataclass(frozen=True, slots=True)
class Artifact:
    filename: str
    url: str
    sha256: str


DEFAULT_ARTIFACTS = (
    Artifact(
        filename="dice-shape.onnx",
        url=f"{MODEL_BASE_URL}/dice-shape.onnx?download=true",
        sha256="e25818c873ca68954952f23d203cc90a9b04f77cbf29ac74a0f24691bb4e16f9",
    ),
    Artifact(
        filename="dice-shape.classes.json",
        url=f"{MODEL_BASE_URL}/dice-shape.classes.json?download=true",
        sha256="c399a002555f7a541c9cb65f1ab09b9e3849e8bb51c232edb74a18cd3a44479a",
    ),
)


class LicenseNotAcceptedError(RuntimeError):
    """Raised when a download is attempted without explicit AGPL acceptance."""


def download_artifacts(
    directory: Path,
    *,
    accept_license: bool,
    artifacts: tuple[Artifact, ...] = DEFAULT_ARTIFACTS,
    timeout_seconds: float = 120,
) -> tuple[Path, ...]:
    """Download pinned model files atomically after explicit license acceptance."""
    if not accept_license:
        raise LicenseNotAcceptedError(
            "model download requires --accept-license; review " + MODEL_CARD_URL
        )
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for artifact in artifacts:
        destination = directory / artifact.filename
        if destination.is_file() and _file_sha256(destination) == artifact.sha256:
            paths.append(destination)
            continue
        _download_artifact(artifact, destination, timeout_seconds=timeout_seconds)
        paths.append(destination)
    return tuple(paths)


def _download_artifact(
    artifact: Artifact,
    destination: Path,
    *,
    timeout_seconds: float,
) -> None:
    if urlsplit(artifact.url).scheme != "https":
        raise ValueError(f"artifact URL must use HTTPS: {artifact.url}")
    request = Request(artifact.url, headers={"User-Agent": "dice-detector/0.1.0"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{artifact.filename}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            digest = sha256()
            # The URL scheme is restricted above; B310 cannot use file/custom URLs.
            with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(f"checksum mismatch for {artifact.filename}")
        temporary_path.chmod(0o644)
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
