"""Build the vendored SciPy 1.11.4 TNC core locally with a configured C compiler."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(source, destination):
    # On Windows moving an object out of a private build directory can retain
    # that directory's restrictive ACL. Stage in the final directory instead.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        staged = Path(stream.name)
        with source.open("rb") as src:
            shutil.copyfileobj(src, stream)
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def main(compiler):
    source = tomllib.loads((ROOT / "source.toml").read_text(encoding="utf-8"))
    for name, expected in source["sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"pinned SciPy source mismatch: {name}")
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if system not in ("linux", "darwin", "windows"):
        raise ValueError(f"unsupported TNC platform: {system}")
    filename = {"linux": "libqhdopt_tnc.so", "darwin": "libqhdopt_tnc.dylib",
                "windows": "qhdopt_tnc.dll"}[system]
    cc = shutil.which(compiler or os.environ.get("CC") or ("cl" if system == "windows" else "cc"))
    if cc is None:
        raise SystemExit("Set CC or pass --cc pointing to a C compiler. For MSVC use a Native Tools prompt.")
    target = ROOT / "build" / f"{system}-{arch}"
    target.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=target) as directory:
        binary = Path(directory) / filename
        sources = [str(ROOT / "tnc.c"), str(ROOT / "abi.c")]
        if Path(cc).stem.lower() == "cl":
            cmd = [cc, "/nologo", "/O2", "/Brepro", "/fp:precise", "/MT", "/LD", *sources,
                   "/link", "/Brepro", "/INCREMENTAL:NO", "/EXPORT:tnc", "/EXPORT:qhdopt_tnc_abi",
                   "/OUT:" + str(binary)]
        else:
            cmd = [cc, "-O2", "-std=c99", "-fPIC", "-ffp-contract=off",
                   "-dynamiclib" if system == "darwin" else "-shared", *sources, "-lm", "-o", str(binary)]
        subprocess.run(cmd, check=True, cwd=directory)
        digest = sha256(binary)
        receipt = Path(directory) / "build.toml"
        receipt.write_text(f'platform = "{system}-{arch}"\ncommit = "{source["commit"]}"\n'
                           f'abi = {source["abi"]}\nsha256 = "{digest}"\n', encoding="utf-8")
        publish(binary, target / filename)
        publish(receipt, target / "build.toml")
        publish(ROOT / "LICENSE", target / "LICENSE")
    print(f"Built {target / filename}  sha256={digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="C compiler executable")
    main(parser.parse_args().cc)
