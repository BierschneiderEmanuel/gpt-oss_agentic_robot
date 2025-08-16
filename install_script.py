#!/usr/bin/env python3
import subprocess
import sys
import os

def read_packages(path):
    """Read package specs from a file, skip blanks & lines starting with ‘#’."""
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out

def install_with_pip(pkgs, dry_run=False):
    """
    Install each package via pip.
    We call: python -m pip install <pkg>
    """
    python_exe = sys.executable
    pip_cmd    = [python_exe, "-m", "pip", "install"]

    for pkg in pkgs:
        cmd = pip_cmd + [pkg]
        print(f"\n→ Installing {pkg!r}:")
        print("  ", " ".join(cmd))
        if dry_run:
            continue

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            print(f"✔ {pkg} OK")
        else:
            print(f"✖ {pkg} failed, skipping")
            # Uncomment to see pip’s error:
            # print(proc.stderr, file=sys.stderr)

if __name__ == "__main__":
    # e.g. `python install_pkgs.py`, or `python install_pkgs.py mylist.txt`
    spec_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    packages = read_packages(spec_file)

    # Set to True if you just want to see the commands without running them
    dry_run = False

    install_with_pip(packages, dry_run=dry_run)