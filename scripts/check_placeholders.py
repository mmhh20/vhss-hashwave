from pathlib import Path
root=Path(__file__).resolve().parents[1]
needles=("REPLACE_WITH", "REPLACE WITH")
hits=[]
for p in root.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".md",".toml",".cff",".yml",".yaml"}:
        text=p.read_text(errors="ignore")
        for i,line in enumerate(text.splitlines(),1):
            if any(n in line for n in needles): hits.append(f"{p.relative_to(root)}:{i}: {line.strip()}")
print("\n".join(hits))
raise SystemExit(1 if hits else 0)
