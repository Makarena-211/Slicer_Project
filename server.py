#from fastapi import FastAPI, File, UploadFile, HTTPException, Request
#from fastapi.responses import JSONResponse, HTMLResponse
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#from pydantic import BaseModel, conlist
from FastSAM.fastsam.model import FastSAM
from FastSAM.fastsam.prompt import FastSAMPrompt

import torch
import numpy as np
#import json
#import sklearn
from sklearn.preprocessing import MinMaxScaler
#import requests
from typing import Any, Dict, AnyStr, List, Union
#from segment_anything.utils.transforms import ResizeLongestSide
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
#from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel

app = FastAPI(
)

security = HTTPBasic()
JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "root"
    correct_password = "1111"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

@app.post("/masks")
async def data_json1(mask: JSONStructure = None, credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    points = mask[b"points"]
    roi = mask[b"roi"]
    pixel_arr = np.array(mask[b"pixel_arr"])
    input_label = np.array(mask[b"input_label"])
    print(f"Данные из json {points, roi, type(input_label)}")
    normalized_rgb_image = normalize_pixel_array(pixel_arr)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    model = FastSAM("FastSAM.pt")
    mask_points, mask_roi = mask_array_all(points, roi, normalized_rgb_image, input_label, model, DEVICE)

    data_mask = {
        "mask_points": mask_points.tolist(),
        "mask_roi": mask_roi.tolist()
    }
    print(len(data_mask["mask_roi"]))



    return data_mask


def normalize_pixel_array(pixel_array):

    scaler = MinMaxScaler()
    # Меняем размерность массива пикселей для работы с MinMaxScaler
    reshaped_pixel_array = pixel_array.reshape(-1, 1)
    normalized_pixel_array = scaler.fit_transform(reshaped_pixel_array)
    # Возвращаем нормализованный массив в исходную форму
    normalized_pixel_array = normalized_pixel_array.reshape(pixel_array.shape)

    # Создайте каналы R, G и B, взяв значения из нормализованного массива
    R_channel = normalized_pixel_array  # Красный канал
    G_channel = normalized_pixel_array  # Зеленый канал (полностью нулевой массив)
    B_channel = normalized_pixel_array  # Синий канал (полностью нулевой массив)

    # Сформируйте массив RGB из трех каналов
    rgb_image = np.stack([R_channel, G_channel, B_channel], axis=-1)

    # Масштабируйте значения обратно к диапазону [0, 255]
    rgb_image = (rgb_image * 255).astype(np.uint8)
    #print(points, roi, rgb_image)
    return rgb_image

def mask_array_all(points, roi, normalized_rgb_image, input_label, model, DEVICE):  # расчитано на то что есть и roi и points
    input_points = points
    input_points = [list(map(round, inner_list)) for inner_list in input_points]
    roi = list(map(int, [item for sublist in roi for item in sublist]))

    print(f"input_label {input_label, type(input_label)}")
    print(f"points {input_points, type(input_points)}")
    print(f"roi {roi, type(roi)}")

    model = model
    IMAGE_PATH = normalized_rgb_image
    DEVICE = DEVICE


    if len(roi) > 0:
        everything_results_roi = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
        prompt_process_roi = FastSAMPrompt(IMAGE_PATH, everything_results_roi, device=DEVICE)
        mask_roi = prompt_process_roi.box_prompt(bbox=roi)
        count_true = np.count_nonzero(mask_roi)
    else:
        mask_roi = np.array([])
    if len(input_points) > 0:
        everything_results_point = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.55, iou=0.75, )
        prompt_process_point = FastSAMPrompt(IMAGE_PATH, everything_results_point, device=DEVICE)
        mask_points = prompt_process_point.point_prompt(points=input_points, pointlabel=input_label)
        count_true = np.count_nonzero(mask_points)
    else:
        mask_points = np.array([])
    print(mask_points, mask_roi, type(mask_points), type(mask_roi))
    print(count_true)

    return mask_points, mask_roi














#uvicorn server:app --reload
#C:\Users\mnfom\Documents\Работа\ML\pythonProject\
