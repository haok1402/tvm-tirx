# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CUDA toolchain selection and native argument lists."""

import re
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import tvm_ffi


class BackendConfig(TypedDict, total=False):
    """Optional CUDA overrides. Toolchain arguments are passed as individual argv items.

    ``fp_mode`` names the floating-point mode instead of spelling it in ``nvcc`` / ``nvrtc``;
    see :func:`fp_mode_flags`. Without it, the argument lists decide, and the default lists pass
    ``--use_fast_math``.
    """

    arch: str
    compiler: Literal["nvcc", "nvrtc"]
    target_format: Literal["ptx", "cubin", "fatbin"]
    fp_mode: str | dict[str, bool]
    nvcc: list[str]
    nvrtc: list[str]
    ptxas: list[str]
    perf_remarks: Literal["warn", "error", "ignore"]


# Floating-point switches of nvcc and NVRTC, by fp_mode key. Both accept the same spellings.
_FP_SWITCHES = {
    "ftz": "--ftz",
    "prec_div": "--prec-div",
    "prec_sqrt": "--prec-sqrt",
    "fmad": "--fmad",
}
_FP_MODE_KEYS = ("fast_math", *_FP_SWITCHES)
# --use_fast_math implies --ftz=true --prec-div=false --prec-sqrt=false --fmad=true, and maps
# math functions to their approximate intrinsics (sinf -> __sinf). "precise" spells out the
# compilers' own defaults, i.e. a build without --use_fast_math.
_FP_MODES = {
    "fast": {"fast_math": True},
    "precise": {
        "fast_math": False,
        "ftz": False,
        "prec_div": True,
        "prec_sqrt": True,
        "fmad": True,
    },
}
_FP_FLAG_NAMES = {
    spelling
    for flag in ("--use_fast_math", *_FP_SWITCHES.values())
    for spelling in (flag, flag[1:])
}


def _fp_flag_name(item):
    return item.split("=", 1)[0]


def _fp_flags_in(args):
    return [item for item in args if _fp_flag_name(item) in _FP_FLAG_NAMES]


def _strip_fp_flags(args):
    """Remove floating-point switches, including a separate ``true`` / ``false`` value."""
    result = []
    skip_value = False
    for item in args:
        if skip_value and item in ("true", "false"):
            skip_value = False
            continue
        skip_value = False
        name = _fp_flag_name(item)
        if name in _FP_FLAG_NAMES:
            skip_value = "=" not in item and name.lstrip("-") != "use_fast_math"
            continue
        result.append(item)
    return result


def _validate_fp_mode(value):
    if isinstance(value, str):
        if value not in _FP_MODES:
            raise ValueError(
                f"CUDA backend_config.fp_mode must be 'fast', 'precise' or a mapping, got {value!r}"
            )
        return value
    if not isinstance(value, Mapping):
        raise TypeError("CUDA backend_config.fp_mode must be 'fast', 'precise' or a mapping")
    for key, flag in value.items():
        if key not in _FP_MODE_KEYS:
            raise ValueError(
                f"Unknown CUDA backend_config.fp_mode key {key!r}; expected one of {_FP_MODE_KEYS}"
            )
        if not isinstance(flag, bool):
            raise TypeError(f"CUDA backend_config.fp_mode[{key!r}] must be a bool")
    return dict(value)


def fp_mode_flags(fp_mode):
    """Compiler arguments for an ``fp_mode`` (the same for nvcc and NVRTC).

    - ``"fast"``: ``--use_fast_math`` (the default lists' setting).
    - ``"precise"``: ``--ftz=false --prec-div=true --prec-sqrt=true --fmad=true``, the
      compilers' defaults, i.e. a library built without fast math. FMA contraction stays on, as
      in nvcc's default; add ``{"fmad": False}`` to turn it off.
    - A mapping of ``fast_math``, ``ftz``, ``prec_div``, ``prec_sqrt`` and ``fmad`` to bools:
      ``--use_fast_math`` if ``fast_math``, then one switch per other key given, which overrides
      what fast math implies. ``{"fast_math": True, "ftz": False}`` is fast math without
      flushing subnormals. An omitted switch keeps the compiler's choice.
    """
    mode = _validate_fp_mode(fp_mode)
    if isinstance(mode, str):
        mode = _FP_MODES[mode]
    flags = ["--use_fast_math"] if mode.get("fast_math", False) else []
    for key, flag in _FP_SWITCHES.items():
        if key in mode:
            flags.append(f"{flag}={'true' if mode[key] else 'false'}")
    return flags


# Only arguments that conflict with TVM's architecture, artifact, or tool routing
# are reserved. General compiler options and their values belong to the compiler.
_RESERVED = {
    "-arch",
    "--gpu-architecture",
    "-gpu-architecture",
    "--gpu-name",
    "-arch-name",
    "-gencode",
    "--generate-code",
    "-code",
    "--gpu-code",
    "-o",
    "--output-file",
    "-optf",
    "--options-file",
    "--cubin",
    "-cubin",
    "--ptx",
    "-ptx",
    "--fatbin",
    "-fatbin",
    "-Xptxas",
    "--ptxas-options",
}


def validate_backend_config(config):
    """Check fixed keys and types, and return independently owned containers."""
    if not isinstance(config, Mapping):
        raise TypeError("CUDA backend_config must be a mapping")
    result = {}
    for name, value in config.items():
        if name not in BackendConfig.__annotations__:
            raise ValueError(f"Unknown CUDA backend_config key: {name!r}")
        if name == "fp_mode":
            result[name] = _validate_fp_mode(value)
        elif name in ("nvcc", "nvrtc", "ptxas"):
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise TypeError(f"CUDA backend_config.{name} must be a sequence of strings")
            if not all(isinstance(v, str) for v in value):
                raise TypeError(f"CUDA backend_config.{name} must be a sequence of strings")
            for item in value:
                if item.split("=", 1)[0] in _RESERVED:
                    raise ValueError(
                        f"{name}: {item!r} is managed by backend_config or the compiler adapter"
                    )
            result[name] = list(value)
        elif not isinstance(value, str):
            raise TypeError(f"CUDA backend_config.{name} must be a string")
        else:
            result[name] = value
    if "arch" in result and not re.fullmatch(r"sm_[0-9]+[af]?", result["arch"]):
        raise ValueError("CUDA backend_config.arch must name a real architecture, e.g. sm_100a")
    if "compiler" in result and result["compiler"] not in ("nvcc", "nvrtc"):
        raise ValueError("CUDA backend_config.compiler must be 'nvcc' or 'nvrtc'")
    if "target_format" in result and result["target_format"] not in ("ptx", "cubin", "fatbin"):
        raise ValueError("CUDA backend_config.target_format must be 'ptx', 'cubin', or 'fatbin'")
    if "perf_remarks" in result and result["perf_remarks"] not in ("warn", "error", "ignore"):
        raise ValueError("CUDA backend_config.perf_remarks must be 'warn', 'error', or 'ignore'")
    if result.get("compiler") == "nvrtc" and result.get("target_format") == "fatbin":
        raise ValueError("NVRTC cannot produce fatbin; use compiler='nvcc'")
    if "fp_mode" in result:
        for name in ("nvcc", "nvrtc"):
            if _fp_flags_in(result.get(name, [])):
                raise ValueError(
                    f"CUDA backend_config sets fp_mode and also floating-point flags in {name} "
                    f"({_fp_flags_in(result[name])}); use one or the other"
                )
    return result


TARGET_SOURCE = "the target's backend_config"
FUNCTION_SOURCE = "the function's backend_config attribute"
DEFAULTS_SOURCE = "the compilation defaults"


def merge_backend_config(
    base, override, *, base_source="the enclosing configuration", override_source="backend_config"
):
    """Overlay one validated CUDA configuration on another; ``override`` is more specific.

    Keys replace inherited keys, and argument lists replace inherited lists. ``fp_mode``
    composes with the lists: an ``fp_mode`` removes the floating-point flags from inherited
    ``nvcc`` / ``nvrtc`` lists and keeps their other arguments, so a kernel's mode survives a
    caller's ``{"nvrtc": ["-lineinfo", ...]}``. Two different explicit modes, or an inherited
    ``fp_mode`` and floating-point flags in a more specific list, are an error: a silent winner
    would hide a precision change. ``base_source`` and ``override_source`` name the two
    configurations in that error.
    """
    result = dict(base)
    inherited = base.get("fp_mode")
    mode = override.get("fp_mode")
    if mode is not None:
        if inherited is not None and fp_mode_flags(inherited) != fp_mode_flags(mode):
            raise ValueError(
                f"Conflicting CUDA fp_mode: {override_source} sets {mode!r}, but {base_source} "
                f"sets {inherited!r}. Set the same mode in both, or drop one."
            )
        for name in ("nvcc", "nvrtc"):
            if name in result and name not in override:
                result[name] = _strip_fp_flags(result[name])
    elif inherited is not None:
        for name in ("nvcc", "nvrtc"):
            if _fp_flags_in(override.get(name, [])):
                raise ValueError(
                    f"Conflicting CUDA floating-point settings: {override_source} sets "
                    f"{_fp_flags_in(override[name])} in {name}, but {base_source} sets "
                    f"fp_mode={inherited!r}. Set fp_mode in {override_source} instead."
                )
    result.update(override)
    return result


def default_backend_config():
    """Return fresh CUDA defaults, preserving the existing compilation policy."""
    return BackendConfig(
        compiler="nvrtc",
        nvcc=["--use_fast_math"],
        nvrtc=["--use_fast_math"],
        ptxas=["-v", "--warn-on-local-memory-usage", "--register-usage-level=10"],
    )


def resolve_backend_config(config=None, target=None, *, source="backend_config"):
    """Resolve one CUDA configuration after applying target/tag defaults.

    ``source`` names ``config`` in an fp_mode conflict with the target's configuration.
    """
    from tvm_tirx.target import Target

    from ..config import copy_backend_config

    overrides = validate_backend_config(config or {})
    if target is None:
        target = Target(
            {"kind": "cuda", **({"arch": overrides["arch"]} if "arch" in overrides else {})}
        )
    else:
        target = Target(target)
    if target.kind.name != "cuda":
        raise ValueError("CUDA backend_config requires a CUDA target")
    result = default_backend_config()
    result = merge_backend_config(
        result,
        copy_backend_config(target.attrs.get("backend_config", {})).get("cuda", {}),
        override_source=TARGET_SOURCE,
    )
    result = merge_backend_config(
        result, overrides, base_source=TARGET_SOURCE, override_source=source
    )
    if "arch" not in result:
        arch = target.attrs.get("arch")
        if arch is None:
            raise ValueError(
                "CUDA compilation requires an explicit backend_config arch or Target arch "
                "when no CUDA device is available"
            )
        result["arch"] = str(arch)
    return validate_backend_config(result)


def compiler_options(config):
    """Forward argv to the selected frontend and its ptxas stage."""
    options = list(config[config["compiler"]])
    if "fp_mode" in config:
        options = fp_mode_flags(config["fp_mode"]) + options
    if config["compiler"] == "nvcc":
        if config["ptxas"]:
            options.append(f"--ptxas-options={','.join(config['ptxas'])}")
    else:
        options.extend(f"--ptxas-options={value}" for value in config["ptxas"])
    return options


@tvm_ffi.register_global_func("cuda.resolve_backend_config")
def _resolve_entry(local, defaults, target, source="", function_config=None):
    """Resolve ``local`` (named by ``source``) over ``defaults`` for ``target``.

    For a function, ``defaults`` are the compilation defaults. For a device entry, ``defaults``
    is its function's resolved configuration and ``function_config`` the function's own
    ``backend_config`` attribute: together they tell where an inherited fp_mode came from, so
    that a conflict error names both sources.
    """
    from tvm_tirx.target import Target

    from ..config import backend_config_json, copy_backend_config, parse_backend_config

    source = source or FUNCTION_SOURCE
    local_config = parse_backend_config(local).get("cuda", {})
    default_config = parse_backend_config(defaults).get("cuda", {})
    defaults_source = DEFAULTS_SOURCE
    if function_config is not None:
        # A device entry: its defaults are the function's resolved configuration.
        target_config = copy_backend_config(Target(target).attrs.get("backend_config", {}))
        if parse_backend_config(function_config).get("cuda", {}).get("fp_mode") is not None:
            defaults_source = FUNCTION_SOURCE
        elif target_config.get("cuda", {}).get("fp_mode") is not None:
            defaults_source = TARGET_SOURCE
    merged = merge_backend_config(
        default_config, local_config, base_source=defaults_source, override_source=source
    )
    merged_source = source if local_config.get("fp_mode") is not None else defaults_source
    config = resolve_backend_config(merged, target, source=merged_source)
    attrs = dict(target.export())
    attrs["arch"] = config["arch"]
    return [Target(attrs), backend_config_json({"cuda": config})]
