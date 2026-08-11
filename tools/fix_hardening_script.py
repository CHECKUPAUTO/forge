from pathlib import Path

p = Path("tools/hardening_migrate.py")
text = p.read_text()
old = r'    "extern \"C\"",'
new = r'    "extern \\\"C\\\"",'
if old not in text:
    raise SystemExit("migration escape pattern not found")
p.write_text(text.replace(old, new, 1))
