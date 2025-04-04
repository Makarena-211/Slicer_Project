from pydantic import BaseModel
from typing import List

class MaskRequest(BaseModel):
    points: List[List[float]]  
    roi: List[float]            
    pixel_arr: List[List[float]] 
    input_label: List[float]     

class MaskResponse(BaseModel):
    mask_points: List[List[float]]
    mask_roi: List[float]