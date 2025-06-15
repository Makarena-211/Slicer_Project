from dotenv import load_dotenv
import os

load_dotenv()

import numpy as np
import uvicorn
import logging
from inference_utils import SegmentAnythingONNX
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from models import MaskRequest, MaskResponse
from contextlib import asynccontextmanager

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
        yield
    except Exception as e:
        logging.error(f"Ошибка инициализации модели: {e}")
        raise HTTPException(status_code=500, detail="Ошибка инициализации модели SAM")
    finally:
        del app.state.model
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
        np.set_printoptions(threshold=np.inf)
        
        # logging.info(f"IMAGE: {normalized_rgb_image}")
        # logging.info(f"IMAGE: {type(normalized_rgb_image)}")
        # logging.info(f"IMAGE: {normalized_rgb_image.shape}")
    except HTTPException as e:
        raise e

    model = app.state.model

    try:
        embedding = model.encode(normalized_rgb_image)
    except Exception as e:
        logging.error(f"Ошибка кодирования изображения: {e}")
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    try:
        masks = await mask_array_all(points=points, roi=roi, input_label=input_label, embedding=embedding, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации масок: {e}")

    return MaskResponse(mask_fiducials=masks.tolist())


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

async def mask_array_all(points: np.ndarray, roi: np.ndarray, input_label: np.ndarray, embedding, model):
    logging.info("Генерация масок")

    print(roi)
    print(points)
    prompt = []
    
    if roi.size > 0:
        prompt = [{"type": "rectangle", "data": roi[0]}]
    elif points.size > 0:
        prompt = [{"type":"point", "data":points[0], "label":input_label[0]}]
    logging.info(f"Prompt: {prompt}")
    try:
        masks = model.predict_masks(embedding, prompt)
        np.save("mask.npy", masks)
    except Exception as e:
        logging.info(f"Error:{e}")
    mask_fiducials = (masks > 0.5).astype(np.uint8)


    return mask_fiducials

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )



