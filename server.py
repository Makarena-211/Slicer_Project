from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pydantic import BaseModel, conlist
import torch
import numpy as np
import json
import sklearn
from sklearn.preprocessing import MinMaxScaler
import requests
from typing import Any, Dict, AnyStr, List, Union

app = FastAPI(
)

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]
'''
class Mask(BaseModel):
    points: List[List[float]]
    roi: List[List[float]]
    pixel_arr: List[List[List[int]]]

@app.get("/")
async def read_root():
    html_content = "<h1>Сервер </h1>"
    return HTMLResponse(content=html_content)
'''
@app.post("/masks")
async def data_json1(mask: JSONStructure = None):
    points = mask[b"points"]
    roi = mask[b"roi"]
    pixel_arr = np.array(mask[b"pixel_arr"])
    first_slice = pixel_arr[0, :, :]
    normalized_points, normalized_roi, normalized_rgb_image = normalize_pixel_array(first_slice, points, roi)
    masks_points, masks_roi = mask_array_all(normalized_points, normalized_roi, normalized_rgb_image)

    #print(normalized_points, normalized_roi, type(normalized_rgb_image))

    data_mask = {
        "mask_points": masks_points.tolist(),
        "mask_roi": masks_roi.tolist()
    }

    json_file_path = r"C:\Users\mnfom\Documents\Работа\ML\pythonProject\masks.json"

    # Открываем файл в режиме записи и сохраняем словарь как JSON
    with open(json_file_path, "w") as json_file:
        json.dump(data_mask, json_file)
    #print(masks_roi)

    return data_mask

def normalize_pixel_array(pixel_array, points, roi):
    #pixel_array, points, roi = ...
    roi = [[round(value, 2) for value in sublist] for sublist in roi]
    points = [[round(elem) for i, elem in enumerate(sublist) if (i + 1) % 3 != 0] for sublist in points]
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
    return points, roi, rgb_image

def mask_array_all(points, roi, rgb_image):  # расчитано на то что есть и roi и points
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"  # загрузка модели
    sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth').to(device=DEVICE)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_image)

    input_roi = np.array(roi)
    input_boxes = torch.tensor(input_roi, device=predictor.device)  # закидываем np.array в тензор, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, rgb_image.shape[:2])  #
    masks_roi, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False)
        #print(masks_roi)

    input_label = np.array([1] * len(points))
    input_point = np.array(points)

    masks_points, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=False)  # если тут true, то выдает 3 маски
        #print(masks_points)
    return masks_roi, masks_points



'''

def save_and_load_masks():
    mask_points, mask_roi = mask_array_all()
    # Сохранение
    data = {
        "mask_points": mask_points.tolist(),
        "mask_roi": mask_roi.tolist()
    }
    with open('mask.json', "w") as json_file:
        json.dump(data, json_file)
    # Загрузка
    with open('mask.json', "r") as json_file:
        loaded_data = json.load(json_file)
    loaded_mask_points_numpy = np.array(loaded_data["mask_points"])
    loaded_mask_roi_numpy = np.array(loaded_data["mask_roi"])
    loaded_mask_points = torch.tensor(loaded_mask_points_numpy)
    loaded_mask_roi = torch.tensor(loaded_mask_roi_numpy)
    return True

'''



#uvicorn server:app --reload
#C:\Users\mnfom\Documents\Работа\ML\pythonProject\
