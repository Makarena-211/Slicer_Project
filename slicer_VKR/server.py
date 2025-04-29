from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import logging
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from models import MaskRequest, MaskResponse
from contextlib import asynccontextmanager

MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
USERNAME = "root"
PASSWORD = "1111"
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Инициализация модели SAM")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.eval()
        app.state.predictor = SamPredictor(sam)
        yield
    except Exception as e:
        logging.error(f"Ошибка инициализации модели SAM: {e}")
        raise HTTPException(status_code=500, detail="Ошибка инициализации модели SAM")
    finally:
        del app.state.predictor, sam
        logging.info("Модель SAM выгружена из памяти")

app = FastAPI(lifespan=lifespan)
security = HTTPBasic()

async def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logging.error(f"Internal Server Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


@app.post("/masks", response_model=MaskResponse)
async def generate_masks(mask: MaskRequest):
    try:
        points = np.array(mask.points)
        roi = np.array(mask.roi)
        pixel_arr = np.array(mask.pixel_arr)
        input_label = np.array(mask.input_label)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Ошибка обработки входных данных: {e}")
    logging.info("Обработка входных данных JSON")
    
    try:
        normalized_rgb_image = await normalize_pixel_array(pixel_arr)
    except HTTPException as e:
        raise e

    predictor = app.state.predictor
    predictor.set_image(normalized_rgb_image)

    try:
        mask_points, mask_roi = await mask_array_all(points, roi, input_label, predictor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации масок: {e}")

    return MaskResponse(mask_points=mask_points.tolist(), mask_roi=mask_roi.tolist())


async def normalize_pixel_array(pixel_array):

    scaler = MinMaxScaler()
    reshaped_pixel_array = pixel_array.reshape(-1, 1)
    normalized_pixel_array = scaler.fit_transform(reshaped_pixel_array)
    normalized_pixel_array = normalized_pixel_array.reshape(pixel_array.shape)

    R_channel = normalized_pixel_array  
    G_channel = normalized_pixel_array  
    B_channel = normalized_pixel_array  

    rgb_image = np.stack([R_channel, G_channel, B_channel], axis=-1)

    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

async def mask_array_all(points: np.ndarray, roi: np.ndarray, input_label: np.ndarray, predictor: SamPredictor):
    logging.info("Генерация масок")
    mask_points, mask_roi = np.array([]), np.array([])
    print(roi)
    
    try:
        if roi.size > 0:
            mask_roi, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=roi,
                multimask_output=False,
            )
        
        if points.size > 0:
            mask_points, _, _ = predictor.predict(
                point_coords=points,
                point_labels=input_label,
                multimask_output=False,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации масок: {e}")

    return mask_points, mask_roi




