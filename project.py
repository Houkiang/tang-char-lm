"""Run project commands from a source checkout without installing packages."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tang_char_lm.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
