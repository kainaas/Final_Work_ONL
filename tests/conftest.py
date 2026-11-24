import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "2D"))
sys.path.insert(0, str(ROOT / "src" / "3D"))
