#from fastapi import FastAPI, File, UploadFile, HTTPException, Request
#from fastapi.responses import JSONResponse, HTMLResponse
from segment_anything import sam_model_registry, SamPredictor
#from pydantic import BaseModel, conlist
# from ultralytics import FastSAM
# from FastSAM.fastsam.prompt import FastSAMPrompt
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
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from MobileSAM.mobile_sam.modeling import sam
import torch
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
    model_type = "vit_h"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model = FastSAM("FastSAM.pt")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(normalized_rgb_image)
    mask_points, mask_roi = mask_array_all(points, roi, input_label, predictor)

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

def mask_array_all(points, roi, input_label, predictor):  # расчитано на то что есть и roi и points
    input_point = points
    input_point = [list(map(round, inner_list)) for inner_list in input_point]
    roi = list(map(int, [item for sublist in roi for item in sublist]))

    print(f"input_label {input_label, type(input_label)}")
    print(f"points {input_point, type(input_point)}")
    print(f"roi {roi, type(roi)}")
    if len(roi) > 0:
        mask_roi, _, _ = predictor.predict(
            point_coords=None,  # координаты точек не нужны
            point_labels=None,  # метки для точек не нужны
            box=np.array(roi),  # сделали двумерный массив
            multimask_output=False,
        )
        count_true = np.count_nonzero(mask_roi)
    else:
        mask_roi = np.array([])
    if len(input_point) > 0:
        mask_points, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=input_label,
            multimask_output=False,  # если тут true, то выдает 3 маски
        )
        count_true = np.count_nonzero(mask_points)
    else:
        mask_points = np.array([])

    print(mask_points, type(mask_points))
    print(count_true)

    return mask_points, mask_roi














#uvicorn server:app --reload
#C:\Users\mnfom\Documents\Работа\ML\pythonProject\
