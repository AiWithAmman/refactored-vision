"""Live check of the configured model provider, before you record anything.

Usage:  python scripts/smoke_test.py photo.jpg keys
Needs the selected provider's API key set and LUMEN_MOCK=0 in .env.
"""
import mimetypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app import models  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    path, label = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        image = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    if models.is_mock():
        print("WARNING: LUMEN_MOCK=1, so nothing below calls Meta. Set LUMEN_MOCK=0 in .env.\n")

    checks = [
        ("SAM 3.1 find_object", lambda: models.find_object(image, label, mime)),
        ("Muse Spark describe_place", lambda: models.describe_place(image, label, mime)),
        ("Muse Spark route", lambda: models.route(f"Where are my {label}?")),
        ("Muse Spark read_text", lambda: models.read_text(image, mime)),
    ]
    ok = True
    for name, fn in checks:
        try:
            print(f"[ok]   {name}: {fn()}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("\nAll checks passed." if ok else "\nFix the failures above first (paste them to your assistant).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
