from flask import Flask, request, jsonify, Response
from functools import wraps
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from segment_anything import sam_model_registry, SamPredictor

from PIL import Image

app = Flask(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = r"sam_vit_h_4b8939.pth"


print("Инициализация моделей...")
mobile_sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)  
mobile_sam_model.to(device=DEVICE)
mobile_sam_model.eval()
sam_predictor = SamPredictor(mobile_sam_model) 
print("Модели успешно инициализированы.")


def normalize_to_grayscale(image_array):
    """
    Нормализует 2D массив пикселей (изображение) в диапазон [0, 255], чтобы представить в grayscale.

    Параметры:
        image_array (numpy.ndarray): Массив размером (высота, ширина).

    Возвращает:
        numpy.ndarray: Нормализованный массив в grayscale с типом uint8.
    """
    # Проверяем, что входной массив является 2D
    if len(image_array.shape) != 2:
        raise ValueError("Входной массив должен быть двумерным (высота, ширина).")
    
    # Нормализация значений
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_image = (image_array - min_val) / (max_val - min_val)

    # Преобразование в формат grayscale
    grayscale_image = (normalized_image * 255).astype(np.uint8)
    
    return grayscale_image  # (512, 512)


def normalize_pixel_array(pixel_array):
    """Нормализация массива пикселей и преобразование в RGB"""
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



def predict_by_points(points, predictor):
    """
    Генерация маски на основе точек.
    
    Параметры:
        points (list): Список координат точек.
        input_label (list): Метки точек (например, 1 для положительных и 0 для отрицательных точек).
        predictor (SamPredictor): Экземпляр предсказателя.
    
    Возвращает:
        numpy.ndarray: Сгенерированная маска на основе точек.
    """
    if len(points) == 0:
        return np.array([])

    input_point = [list(map(round, inner_list)) for inner_list in points]
    mask_points, scores, logits = predictor.predict(
        point_coords=np.array(input_point),
        point_labels=[1],
        multimask_output=False,
    )
    return mask_points


def predict_by_roi(roi, predictor):
    """
    Генерация маски на основе ROI.
    
    Параметры:
        roi (list): Список координат области интереса [x_min, y_min, x_max, y_max].
        predictor (SamPredictor): Экземпляр предсказателя.
    
    Возвращает:
        numpy.ndarray: Сгенерированная маска на основе ROI.
    """
    if len(roi) == 0:
        return np.array([])

    roi_array = list(map(int, [item for sublist in roi for item in sublist]))
    mask_roi, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(roi_array),
        multimask_output=False,
    )
    return mask_roi



def save_mask_as_image(mask, output_path="mask.png"):
    """
    Сохраняет бинарный массив как изображение.

    Параметры:
        mask (numpy.ndarray): Бинарный массив размером (1, H, W) или (H, W).
        output_path (str): Путь для сохранения изображения.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # Преобразуем (1, H, W) в (H, W)

    # Масштабируем маску в диапазон [0, 255]
    mask_image = (mask * 255).astype(np.uint8)
    

    # Создаем изображение и сохраняем
    img = Image.fromarray(mask_image)
    img.save(output_path)
    print(f"Маска сохранена как {output_path}")



@app.route("/masks/points", methods=["POST"])
def create_mask_by_points():
    """
    Генерация маски на основе точек.
    """
    try:
        # Получение данных из JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        points = data.get("points")
        arr = np.array(data.get("pixel_arr"))

        if points is None or arr is None:
            return jsonify({"error": "Missing required fields: 'points' or 'pixel_arr'"}), 400
        print(points, arr)
        try:
            pixel_arr =  normalize_to_grayscale(arr)

            # Нормализация изображения
            normalized_rgb_image = normalize_pixel_array(pixel_arr)

            # Установка изображения для предиктора
            sam_predictor.set_image(normalized_rgb_image)

            # Генерация маски по точкам
            mask_points = predict_by_points(points, sam_predictor)

            save_mask_as_image(mask_points, "binary_mask.png")
        except Exception as e:
            print(e)
        # Формирование ответа
        return jsonify({"mask_points": mask_points.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/masks/roi", methods=["POST"])
def create_mask_by_roi():
    """
    Генерация маски на основе ROI.
    """
    try:
        # Получение данных из JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        roi = data.get("roi")
        arr = np.array(data.get("pixel_arr"))

        if roi is None or arr is None:
            return jsonify({"error": "Missing required fields: 'roi' or 'pixel_arr'"}), 400


        pixel_arr =  normalize_to_grayscale(arr)
        # Нормализация изображения
        normalized_rgb_image = normalize_pixel_array(pixel_arr)


        # Установка изображения для предиктора
        sam_predictor.set_image(normalized_rgb_image)

        # Генерация маски по ROI
        mask_roi = predict_by_roi(roi, sam_predictor)

        # Формирование ответа
        return jsonify({"mask_roi": mask_roi.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

