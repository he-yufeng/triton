"""Regression tests for empty AOT signatures (#12507)."""

from pathlib import Path

import pytest

from triton.tools.link import (
    HeaderParser,
    gen_signature,
    gen_signature_with_full_args,
    make_algo_decls,
    make_default_algo_kernel,
    make_func_pointers,
    make_global_decl,
    make_kernel_hints_dispatcher,
    make_kernel_meta_const_dispatcher,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent


class _EmptyMeta:
    orig_kernel_name = "all_constexpr"
    arg_names = []
    arg_ctypes = []
    sizes = []
    num_specs = 0
    sig_hash = "hash"
    suffix = ""


class _MixedMeta:
    orig_kernel_name = "mixed"
    arg_names = ["x", "n"]
    arg_ctypes = ["float*", "int32_t"]
    sizes = [None, None]
    num_specs = 0


def test_aot_templates_do_not_add_an_unconditional_comma():
    for path in (
        REPO_ROOT / "third_party/nvidia/tools/cuda/compile.c",
        REPO_ROOT / "third_party/nvidia/tools/cuda/compile.h",
        REPO_ROOT / "third_party/amd/tools/hip/compile.c",
        REPO_ROOT / "third_party/amd/tools/hip/compile.h",
    ):
        content = path.read_text()
        assert "stream, {signature}" not in content, path
        assert "stream{signature}" in content, path


def test_empty_linker_signature_is_empty_and_joined_once():
    assert gen_signature(_EmptyMeta()) == "TT_StreamTy stream"
    assert gen_signature_with_full_args(_EmptyMeta()) == "TT_StreamTy stream"
    assert gen_signature(_MixedMeta()) == "TT_StreamTy stream, float* x, int32_t n"
    assert gen_signature_with_full_args(_MixedMeta()) == "TT_StreamTy stream, float* x, int32_t n"


def test_empty_linker_declaration_and_call_have_no_trailing_comma():
    meta = _EmptyMeta()
    declaration = make_global_decl(meta)
    assert "TT_ResultTy all_constexpr_default(TT_StreamTy stream);" in declaration
    assert "TT_ResultTy all_constexpr(TT_StreamTy stream, int algo_id);" in declaration
    assert "all_constexpr(stream, 0);" in make_default_algo_kernel(meta)
    assert "all_constexpr_kernels[algo_id](stream);" in make_kernel_meta_const_dispatcher(meta)
    assert "(*kernel_func_t)(TT_StreamTy stream)" in make_func_pointers(["all_constexpr_hash"], meta)
    assert "TT_ResultTy all_constexpr(TT_StreamTy stream);" in make_algo_decls("all_constexpr", [meta])
    assert "all_constexpr_hash_(TT_StreamTy stream);" in make_kernel_hints_dispatcher("all_constexpr", [meta])


def test_empty_linker_header_parses_to_zero_arguments():
    parser = HeaderParser()
    parser.extract_linker_meta("// tt-linker: all_constexpr_hash_::1_warps1xstages3\n// tt-linker-backend: cuda\n")
    kernel = parser.kernels["all_constexpr_1_warps1xstages3"][0]
    assert kernel.arg_names == []
    assert kernel.arg_ctypes == []
    assert kernel.sizes == []
