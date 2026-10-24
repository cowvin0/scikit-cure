"""Check versions without importing the source package or optional backends."""

from pathlib import Path
import re


def check_version(root: Path) -> str:
    metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
    source = (root / "skcure/__init__.py").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    patterns = ((metadata, r'^version = "([^"]+)"$'),
                (source, r'^__version__ = "([^"]+)"$'),
                (citation, r'^version: (\S+)$'))
    versions = []
    for content, pattern in patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing version matching {pattern}.")
        versions.append(match.group(1))
    if len(set(versions)) != 1:
        raise ValueError(f"Inconsistent package/source/citation versions: {versions}.")
    return versions[0]


if __name__ == "__main__":
    print(f"Consistent version: {check_version(Path(__file__).resolve().parents[1])}")
