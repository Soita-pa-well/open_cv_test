import numpy as np
import cv2
import webcolors
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from fastapi import HTTPException
from typing import Tuple, Optional


class ColorTransformer:
    """
    Класс для обработки и преобразования цветов изображений.
    """
    def __init__(self, num_clusters: int = 5):
        self.num_clusters: int = num_clusters
        self.color_dict: dict = mcolors.CSS4_COLORS

    def most_common_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Определяет наиболее распространённый цвет на изображении.
        """
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.num_clusters)
            kmeans.fit(pixels)
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_
            cluster_counts = np.bincount(cluster_labels)
            most_common_cluster = np.argmax(cluster_counts)
            most_common_color = cluster_centers[most_common_cluster]
            most_common_color_tuple = tuple(map(int, most_common_color))

            return most_common_color_tuple

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка: {e}")

    def closest_color(self, color: Tuple[int, int, int]) -> Optional[str]:
        """
        Находит ближайшее название цвета если не произошло
        точного попадания в цыет.
        """
        closest_name: Optional[str] = None
        min_distance: float = float('inf')

        for name, hex_value in self.color_dict.items():
            r, g, b = webcolors.hex_to_rgb(hex_value)
            distance = np.sqrt((r - color[0])**2 + (
                g - color[1])**2 + (b - color[2])**2)
            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name

    def rgb_to_name(self, rgb_tuple: Tuple[int, int, int]) -> str:
        """
        Преобразует  RGB значение в название цвета. Если точное совпадение не
        найдено, то возвращает ближайшее название цвета.
        """
        try:
            color = webcolors.rgb_to_name(rgb_tuple)
        except ValueError:
            color = self.closest_color(rgb_tuple)
        return color
