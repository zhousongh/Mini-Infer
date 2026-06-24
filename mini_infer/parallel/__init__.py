"""mini_infer.parallel — 分布式扩展：TP、EP、Replica、PP。"""

from .pp_engine import PPEngine
from .replica_engine import ReplicaEngine

__all__ = ["ReplicaEngine", "PPEngine"]

try:
    from .ep_engine import EPEngine
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.ep_engine":
        raise
else:
    __all__.append("EPEngine")
