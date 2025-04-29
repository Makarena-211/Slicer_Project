from pydantic import BaseModel, field_validator
from typing import List, Union
import numpy as np

class MaskRequest(BaseModel):
    points: Union[List[List[float]], List[List[int]], List] = []
    roi: Union[List[List[float]], List[List[int]], List] = []
    pixel_arr: List[List[int]]
    input_label: List[int] = []
    
    @field_validator('points', 'roi', 'pixel_arr', 'input_label')
    def convert_to_numpy(cls, value):
        """Конвертирует списки в numpy массивы и проверяет типы"""
        if isinstance(value, list):
            arr = np.array(value)
            if arr.size > 0:
                return arr
        return np.array([])

class MaskResponse(BaseModel):
    mask_fiducials: Union[List[List[List[bool]]], List]
    
    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """Создает ответ из numpy массива"""
        if mask.ndim == 3:
            return cls(mask_fiducials=mask.astype(bool).tolist())
        return cls(mask_fiducials=[])