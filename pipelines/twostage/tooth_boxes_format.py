from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ToothBoxesRecord:
    """
    One image's tooth detection results.

    Conventions:
    - boxes: XYXY in absolute pixel coordinates on the *original image*.
    - scores: confidence score per box in [0, 1].
    - image_id: COCO image id (int). Optional if only using file_name lookup.
    - file_name: image file basename or relative path inside COCO images folder.
    """

    file_name: str
    boxes: List[List[float]]
    scores: List[float]
    image_id: Optional[int] = None
    model: Optional[str] = None
    meta: Optional[Dict] = None

    def validate(self) -> None:
        if len(self.boxes) != len(self.scores):
            raise ValueError("len(boxes) must equal len(scores)")
        for b in self.boxes:
            if len(b) != 4:
                raise ValueError("Each box must have 4 numbers [x1,y1,x2,y2]")

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        return boxes, scores


def _ensure_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def write_jsonl(records: Sequence[ToothBoxesRecord], path: Union[str, Path]) -> None:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            rec.validate()
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def iter_jsonl(path: Union[str, Path]) -> Iterator[ToothBoxesRecord]:
    path = _ensure_path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield ToothBoxesRecord(**obj)


def load_jsonl_to_index(
    path: Union[str, Path],
    key: str = "file_name",
) -> Dict[Union[str, int], ToothBoxesRecord]:
    """
    Build an in-memory index for quick lookup.
    key: 'file_name' or 'image_id'
    """
    if key not in {"file_name", "image_id"}:
        raise ValueError("key must be one of: file_name, image_id")
    index: Dict[Union[str, int], ToothBoxesRecord] = {}
    for rec in iter_jsonl(path):
        k = getattr(rec, key)
        if k is None:
            raise ValueError(f"Record missing key field: {key}")
        index[k] = rec
    return index

