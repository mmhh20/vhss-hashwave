from __future__ import annotations
import hashlib, json, pathlib, shutil, subprocess, sys

root = pathlib.Path(__file__).resolve().parents[1]
commands = [
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    [sys.executable, "-m", "hashwave", "demo", "--generations", "8", "--beam", "8", "--per-parent", "12", "--seed", "release", "--json"],
]
for cmd in commands:
    subprocess.run(cmd, cwd=root, check=True)
for path in root.rglob("__pycache__"):
    if path.is_dir():
        shutil.rmtree(path)
ignore_dirs = {".git", "build", "rendered", "vhss_hashwave.egg-info"}
manifest = {}
for file in sorted(root.rglob("*")):
    rel = file.relative_to(root)
    if not file.is_file() or any(part in ignore_dirs for part in rel.parts):
        continue
    if file.name in {"SHA256SUMS.json", ".coverage"} or file.suffix == ".pyc":
        continue
    manifest[str(rel)] = hashlib.sha256(file.read_bytes()).hexdigest()
(root / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"verified {len(manifest)} files")
