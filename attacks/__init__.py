"""Paquete de ataques: Adv1 (FDI sensor), Adv2 (datos/modelo) y escenarios coordinados."""
from .adv1_fdi import apply_fdi  # noqa: F401
from .adv2_data import apply_label_flip  # noqa: F401
from .adv2_model import make_model_poison  # noqa: F401
from .coordinated import build_scenario, SCENARIOS  # noqa: F401
