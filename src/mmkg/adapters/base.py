from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from mmkg.schemas.core import GraphDocument

class BaseAdapter(ABC):
    @abstractmethod
    def extract(
        self,
        doc_id: str,
        text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> GraphDocument:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
