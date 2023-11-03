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
from segment_anything.utils.transforms import ResizeLongestSide


app = FastAPI(
)

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]

@app.post("/masks")
async def data_json1(mask: JSONStructure = None):
    points = mask[b"points"]
    roi = mask[b"roi"]
    pixel_arr = np.array(mask[b"pixel_arr"])
    input_label = np.array(mask[b"input_label"])
    print(f"Данные из json {pixel_arr.shape}")
    normalized_rgb_image = normalize_pixel_array(pixel_arr)
    masks_points, masks_roi = mask_array_all(points, roi, normalized_rgb_image, input_label)

    data_mask = {
        "mask_points": masks_points.tolist(),
        "mask_roi": masks_roi.tolist()
    }
    print(len(data_mask["mask_roi"]))

    json_file_path = r"C:\Users\mnfom\Documents\Работа\ML\pythonProject\masks.json"

    #Открываем файл в режиме записи и сохраняем словарь как JSON
    with open(json_file_path, "w") as json_file:
        json.dump(data_mask, json_file)
    print(masks_roi)

    return data_mask

class SAMModelSingleton:
    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model is None:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = "vit_h"  # загрузка модели
            self.model = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth').to(device=DEVICE)

        return self.model

sam_model_singleton = SAMModelSingleton()

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

def mask_array_all(points, roi, rgb_image, input_label):  # расчитано на то что есть и roi и points
    sam = sam_model_singleton.get_model()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_image)

    if len(roi) == 1:
        input_roi = np.array(roi)
        masks_roi, _, _ = predictor.predict(
            point_coords=None,  # координаты точек не нужны
            point_labels=None,  # метки для точек не нужны
            box=input_roi[None, :],  # сделали двумерный массив
            multimask_output=False)
        count_true = np.count_nonzero(masks_roi)
        print(masks_roi)
        print(f"используется 1 ROI {count_true}")

    elif len(roi) > 1:
        input_roi = np.array(roi)
        input_boxes = torch.tensor(input_roi, device=predictor.device)  # закидываем np.array в тензор, device=predictor.device)
        transformed_boxes = resize_transform.apply_boxes_torch(input_boxes, rgb_image.shape[:3]) #
        masks_roi, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        count_true = np.count_nonzero(masks_roi)
        print(masks_roi)
        print(f"используется 2< ROI {count_true}")


    else:
        masks_roi = np.array([])
    if points:
        input_point = np.array(points)
        print(f"input point {input_point}, {type(input_point)}, {input_point.shape}")
        masks_points, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False)  # если тут true, то выдает 3 маски
        print(masks_points)
        count_true = np.count_nonzero(masks_points)
        print(f"кол-во true {count_true}")
    else:
        masks_points = np.array([])

    return masks_points, masks_roi






#uvicorn server:app --reload
#C:\Users\mnfom\Documents\Работа\ML\pythonProject\
