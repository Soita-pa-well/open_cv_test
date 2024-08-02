from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import webcolors
import matplotlib.colors as mcolors
from src.models import ColorName
from sklearn.cluster import KMeans

app = FastAPI()


def most_common_color(image, num_clusters=5):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    cluster_counts = np.bincount(cluster_labels)
    most_common_cluster = np.argmax(cluster_counts)
    most_common_color = cluster_centers[most_common_cluster]
    colors_tuple = tuple(map(int, most_common_color))
    return colors_tuple


def closest_color(color):
    min_colors = {}
    color_dict = mcolors.CSS4_COLORS
    for name, hex_value in color_dict.items():
        red, green, blue = webcolors.hex_to_rgb(hex_value)
        red_square = ((np.int32(red) - np.int32(color[0])) ** 2)
        green_square = ((np.int32(green) - np.int32(color[1])) ** 2)
        blue_square = ((np.int32(blue) - np.int32(color[2])) ** 2)
        min_colors[(red_square + green_square + blue_square)] = name
    closest_color = min_colors[min(min_colors.keys())]

    return closest_color


def rgb_to_name(rgb_tuple):
    try:
        color = webcolors.rgb_to_name(rgb_tuple)
        return color
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
