from pydantic import BaseModel
from typing import List, Union

class MaskRequest(BaseModel):
    points: Union[List[List[float]], List[List[int]], List] 
    roi: Union[List[List[float]], List[List[int]], List]
    pixel_arr: List[List[int]]
    input_label: List[int]     

class MaskResponse(BaseModel):
    mask_fiducials: Union[List[List[List[bool]]], List]
