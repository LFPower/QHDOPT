"""Install the supplied, hash-pinned private beta in a separate Python 3.12 venv."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "phisolve-0.3.1b20261115221654-py3-none-any.whl"
WHEEL_SHA256 = "9f160bf40e18ec916bd298be68322bc86344a007640e6605bc067847a11d9b03"


def wheel_bytes(source):
    """Read only the release wheel; never execute or extract bundled scripts."""
    if source.is_dir():
        data = (source / "packages" / WHEEL_NAME).read_bytes()
    elif source.name == WHEEL_NAME:
        data = source.read_bytes()
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            matches = [item for item in archive.infolist()
                       if item.filename.endswith("/packages/" + WHEEL_NAME)
                       or item.filename == "packages/" + WHEEL_NAME]
            if len(matches) != 1:
                raise ValueError("The archive must contain exactly one expected beta wheel.")
            data = archive.read(matches[0])
    else:
        raise ValueError("Supply the beta ZIP, extracted bundle directory, or release wheel.")
    if hashlib.sha256(data).hexdigest() != WHEEL_SHA256:
        raise ValueError("Wheel SHA256 does not match private beta 20261115221654.")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="beta ZIP, extracted bundle directory, or wheel")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-openphisolve-beta")
    parser.add_argument("--verify-only", action="store_true", help="check the supplied wheel without installing")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Run this installer with Python 3.12.")
    data = wheel_bytes(args.bundle.resolve())
    if args.verify_only:
        print(f"Verified {WHEEL_NAME}: {WHEEL_SHA256}")
        return
    destination = args.venv.resolve()
    if destination in {Path(sys.prefix).resolve(), ROOT / ".venv-openphisolve"}:
        parser.error("Choose a separate beta environment, not the running or legacy environment.")
    if destination.exists() and not (destination / "pyvenv.cfg").is_file():
        parser.error("The destination exists but is not a virtual environment.")
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "-r",
                    str(ROOT / "python" / "requirements-private-beta-cpu.txt")], check=True)
    with tempfile.TemporaryDirectory(prefix="qhdopt-beta-") as temporary:
        wheel = Path(temporary) / WHEEL_NAME
        wheel.write_bytes(data)
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                        "--force-reinstall", str(wheel)], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    verification = (
        "import runpy,sys; w=runpy.run_path(sys.argv[1]); "
        "p=w['verify_source_profile'](); "
        "assert p['source_revision']=='private-beta-20261115221654',p; print(p)"
    )
    subprocess.run([str(python), "-I", "-B", "-c", verification,
                    str(ROOT / "python" / "openphisolve_worker.py")], check=True)
    print(f"Beta Python: {python}")
    print(f'Run: julia --startup-file=no --project=.local/julia-env examples/native_beta.jl "{python}"')


if __name__ == "__main__":
    main()
