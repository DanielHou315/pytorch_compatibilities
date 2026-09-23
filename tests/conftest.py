import sys
from pathlib import Path

# The scripts/ modules are imported directly by the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
