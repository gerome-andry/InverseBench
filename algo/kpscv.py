# algo/kpscv.py -- KPS with the cross-validated ridge fit. See ridge_slr/kps_cv.py.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ridge_slr.kps_cv import KPSCVAlgo  # noqa: F401,E402
