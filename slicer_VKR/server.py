from dotenv import load_dotenv
import os
import numpy as np
import logging
from inference_utils import SegmentAnythingONNX
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

load_dotenv()

# Модели для работы с разными слоями
class SliceData:
    def __init__(self):
        self.embeddings = {}  # {slice_index: embedding}
        self.masks = {}       # {slice_index: mask}

# Модели Pydantic
class MaskRequest(BaseModel):
    slice_index: int
    points: List[List[float]] = []
    roi: List[List[float]] = []
    pixel_arr: List[List[float]]
    input_label: List[int] = []

class MaskResponse(BaseModel):
    slice_index: int
    mask_fiducials: List[List[int]]

# Конфигурация
ENCODER_MODEL_PATH = os.getenv("ENCODER_MODEL_PATH")
DECODER_MODEL_PATH = os.getenv("DECODER_MODEL_PATH")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Инициализация SAM ONNX модели")
    try:
        model = SegmentAnythingONNX(
            encoder_model_path=ENCODER_MODEL_PATH,
            decoder_model_path=DECODER_MODEL_PATH,
        )
        app.state.model = model
        app.state.slice_data = SliceData()  # Хранилище данных по слоям
        yield
    except Exception as e:
        logging.error(f"Ошибка инициализации модели: {e}")
        raise HTTPException(status_code=500, detail="Ошибка инициализации модели SAM")
    finally:
        del app.state.model, app.state.slice_data
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

@app.post("/masks", response_model=MaskResponse)
async def generate_masks(mask: MaskRequest):
    try:
        # Преобразование входных данных
        points = np.array(mask.points, dtype=np.float32)
        roi = np.array(mask.roi, dtype=np.float32)
        pixel_arr = np.array(mask.pixel_arr, dtype=np.float32)
        input_label = np.array(mask.input_label, dtype=np.int32)
        slice_index = mask.slice_index
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Ошибка обработки входных данных: {e}")

    # Нормализация изображения
    try:
        normalized_rgb_image = await normalize_pixel_array(pixel_arr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = app.state.model
    slice_data = app.state.slice_data

    # Кэшируем эмбеддинги для каждого слоя
    if slice_index not in slice_data.embeddings:
        try:
            embedding = model.encode(normalized_rgb_image)
            slice_data.embeddings[slice_index] = embedding
        except Exception as e:
            logging.error(f"Ошибка кодирования изображения: {e}")
            raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    # Генерация масок
    try:
        masks = await mask_array_all(
            points=points,
            roi=roi,
            input_label=input_label,
            embedding=slice_data.embeddings[slice_index],
            model=model
        )
        slice_data.masks[slice_index] = masks  # Сохраняем маску для слоя
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации масок: {e}")

    return MaskResponse(
        slice_index=slice_index,
        mask_fiducials=masks.astype(int).tolist()
    )

async def normalize_pixel_array(pixel_array: np.ndarray) -> np.ndarray:
    """Нормализация изображения и преобразование в RGB"""
    if pixel_array.size == 0:
        raise ValueError("Пустой массив пикселей")
    
    try:
        # Нормализация к [0, 1]
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max - p_min > 0:
            normalized = (pixel_array - p_min) / (p_max - p_min)
        else:
            normalized = np.zeros_like(pixel_array)
        
        # Преобразование в RGB (3 канала)
        return np.stack([normalized]*3, axis=-1).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Ошибка нормализации: {str(e)}")

async def mask_array_all(
    points: np.ndarray,
    roi: np.ndarray,
    input_label: np.ndarray,
    embedding: np.ndarray,
    model: SegmentAnythingONNX
) -> np.ndarray:
    """Генерация масок по точкам и ROI"""
    prompts = []
    
    # Обработка ROI (прямоугольники)
    if roi.size > 0:
        if roi.ndim == 2 and roi.shape[1] == 4:
            for rect in roi:
                prompts.append({
                    "type": "rectangle",
                    "data": rect.tolist()
                })
        else:
            logging.warning(f"Некорректная форма ROI: {roi.shape}")

    # Обработка точек
    if points.size > 0:
        if points.ndim == 2 and points.shape[1] == 2:
            for point, label in zip(points, input_label):
                prompts.append({
                    "type": "point",
                    "data": point.tolist(),
                    "label": int(label)
                })
        else:
            logging.warning(f"Некорректная форма точек: {points.shape}")

    if not prompts:
        raise ValueError("Не заданы корректные точки или ROI")

    try:
        masks = model.predict_masks(embedding, prompts)
        return (masks > 0.5).astype(np.uint8)[0, 0]  # Берем первую маску
    except Exception as e:
        logging.error(f"Ошибка предсказания: {str(e)}")
        raise