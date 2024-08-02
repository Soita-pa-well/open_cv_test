from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from typing import Tuple
import webcolors
import matplotlib.colors as mcolors
from src.models import ColorName

app = FastAPI()


def most_common_color(image: np.ndarray) -> Tuple[int, int, int]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    color_count = {}
    for pixel in pixels:
        color = tuple(pixel)
        if color in color_count:
            color_count[color] += 1
        else:
            color_count[color] = 1
    most_common_color = max(color_count, key=color_count.get)

    return most_common_color


def closest_color(requested_color):
    min_colors = {}
    color_dict = mcolors.CSS4_COLORS
    print(color_dict)
    for name, hex_value in color_dict.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]


def rgb_to_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        return closest_color(rgb_tuple)


@app.post("/uploadimage/")
async def create_upload_file(file: UploadFile = File(...)):
    image = await file.read()
    np_arr = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    closest_name = rgb_to_name(most_common_color(image))
    correct_color = getattr(ColorName, closest_name.upper(), closest_name)
    return {"Общий цвет фото": correct_color.value}
