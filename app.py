from typing import Any, Tuple
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.models import ColorName
from src.main import ColorTransformer

app = FastAPI()


color_processor = ColorTransformer()


@app.post("/uploadimage/")
async def create_upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    """
    Загрузка изображения в формате .jpeg, .png. Доступна на
    http://127.0.0.1:8000/docs#/default/create_upload_file_uploadimage__post
    """
    try:
        image_data: bytes = await file.read()
        np_arr: np.ndarray = np.frombuffer(image_data, dtype=np.uint8)
        image: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400,
                                detail="Формат изображения некорректный")

        common_color: Tuple[int, int, int] = color_processor.most_common_color(
            image)
        closest_name: str = color_processor.rgb_to_name(common_color)
        correct_color: Any = getattr(ColorName,
                                     closest_name.upper(),
                                     closest_name)
        return {"Цвет фото": correct_color.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {e}")
