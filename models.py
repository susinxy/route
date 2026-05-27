"""数据模型与输入解析"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Box:
    idx: int      # 1-indexed
    w: float
    h: float


class Problem:
    """解析后的布局问题"""

    def __init__(self, data: Dict[str, Any]):
        self.n = len(data["box_size"])
        self.boxes = [Box(i + 1, s[0], s[1]) for i, s in enumerate(data["box_size"])]
        self.widths = [s[0] for s in data["box_size"]]
        self.heights = [s[1] for s in data["box_size"]]

        self.sym_x_groups: List[Dict] = data.get("symmetry_x", [])
        self.sym_y_groups: List[Dict] = data.get("symmetry_y", [])

        align = data.get("align", {})
        self.align_left: List[List[int]] = align.get("left", [])
        self.align_right: List[List[int]] = align.get("right", [])
        self.align_top: List[List[int]] = align.get("top", [])
        self.align_bottom: List[List[int]] = align.get("bottom", [])

        self.repeat_groups: List[List[List[int]]] = [
            rg["groups"] for rg in data.get("repeat_groups", [])
        ]

        self.nets: List[List[int]] = data.get("nets", [])
