import hashlib
import pytest
from shopping_grpo.training.grpo.dependency_provenance import verify_python_source

REVISION = "pinned"


def fixture(tmp_path):
    (tmp_path / "__init__.py").write_text("version = 1")
    (tmp_path / "model.py").write_text("model = 2")
    return {"revision": REVISION,
            "archive_url": f"https://codeload.github.com/huggingface/transformers/tar.gz/{REVISION}",
            "package_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in tmp_path.glob("*.py")}}


def test_source_tree_equality_without_fabricating_vcs_metadata(tmp_path):
    manifest = fixture(tmp_path)
    assert verify_python_source(manifest, tmp_path, REVISION) == 2


@pytest.mark.parametrize("change", ["modified", "extra", "missing", "revision"])
def test_source_mismatch_rejected(tmp_path, change):
    manifest = fixture(tmp_path)
    if change == "modified":
        (tmp_path / "model.py").write_text("different")
    elif change == "extra":
        (tmp_path / "extra.py").write_text("extra")
    elif change == "missing":
        (tmp_path / "model.py").unlink()
    else:
        manifest["revision"] = "wrong"
    with pytest.raises(ValueError):
        verify_python_source(manifest, tmp_path, REVISION)
