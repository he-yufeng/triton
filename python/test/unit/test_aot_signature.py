"""Tests for AOT compiler signature generation (#12507)."""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_empty_signature_no_trailing_comma():
    """A kernel with only constexpr args must not produce a trailing comma."""
    cuda_template = REPO_ROOT / "third_party" / "nvidia" / "tools" / "cuda" / "compile.c"
    hip_template = REPO_ROOT / "third_party" / "amd" / "tools" / "hip" / "compile.c"

    for template_path in [cuda_template, hip_template]:
        content = template_path.read_text()
        assert "stream, {signature}" not in content, f"{template_path.name} has trailing comma pattern"
        assert "{signature}" in content, f"{template_path.name} missing signature placeholder"


def test_compile_py_generates_full_signature():
    """compile.py must include the stream parameter in the generated signature."""
    compile_py = REPO_ROOT / "python" / "triton" / "tools" / "compile.py"
    content = compile_py.read_text()
    assert '("stream, " if sig_args else "stream")' in content, "compile.py missing empty-signature handling"
    assert '("stream, " if full_args else "stream")' in content, "compile.py missing full-signature handling"
