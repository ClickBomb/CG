import numpy as np
from numpy import cos, sin, pi
from typing import List, Tuple
from PIL import Image as im


class MyImage:
    def __init__(self, height: int, width: int, pixels: List = None):
        self.height = height
        self.width = width
        if pixels is None:
            self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            self.pixels = np.array(pixels, dtype=np.uint8)
            if len(self.pixels.shape) != 3 or self.pixels.shape[2] != 3:
                assert ValueError("3 channels must be in each pixel!")

    def set_pixel(self, x: int, y: int, color: Tuple[int, int, int]):
        if self.point_exist(x, y):
            self.pixels[y, x] = color
        else:
            assert f"Point ({x}, {y}) doesn't exist"

    def show(self, window_name: str = 'my image'):
        data = im.fromarray(self.pixels)
        data.show(window_name)

    def save(self, filename: str):
        data = im.fromarray(self.pixels)
        data.save(filename)

    def clear(self):
        self.pixels = np.zeros_like(self.pixels)

    def point_exist(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            return True
        return False


def line_1(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int], point_num: int = 100):
    t_arr = np.linspace(0, 1, point_num, endpoint=True)
    for t in t_arr:
        x = int(x0 * (1.0 - t) + x1 * t)
        y = int(y0 * (1.0 - t) + y1 * t)
        img.set_pixel(x, y, color)


def line_2(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1.0 - t) + y1 * t)
        img.set_pixel(x, y, color)


def line_3(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    step = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        step = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1 + 1):
        t = (x - x0) / (x1 - x0)
        y = int(y0 * (1.0 - t) + y1 * t)
        img.set_pixel(y, x, color) if step else img.set_pixel(x, y, color)

def line_4(x0: int, y0: int, x1: int, y1: int, img: MyImage, color: Tuple[int, int, int]):
    step = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        step = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx != 0:
        d_error = abs(dy / dx)
    else:
        d_error = 0
    error = 0.0
    y = y0

    for x in range(x0, x1 + 1):
        if step:
            img.set_pixel(y, x, color)
        else:
            img.set_pixel(x, y, color)
        error += d_error
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1.0


def test_lines():
    h, w = 200, 200
    image = MyImage(h, w)
    color = (255, 255, 255)
    x_mid, y_mid = 100, 100
    alpha = 2 * pi / 13

    points_x, points_y = [], []
    for i in range(0, 13):
        points_x.append(int(100 + 95 * cos(alpha * i)))
        points_y.append(int(100 + 95 * sin(alpha * i)))

    for x, y in zip(points_x, points_y):
        line_1(x_mid, y_mid, x, y, image, color, point_num=50)
    image.show('first method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_2(x_mid, y_mid, x, y, image, color)
    image.show('second method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_3(x_mid, y_mid, x, y, image, color)
    image.show('third method')
    image.clear()

    for x, y in zip(points_x, points_y):
        line_4(x_mid, y_mid, x, y, image, color)
    image.show('fourth method')
    image.clear()


def read_obj(file_path: str):
    vertices = []
    faces = []

    with open(file_path) as file:
        for line in file.readlines():
            data = line.split(' ')
            if data[0] == 'v':
                x, y, z = float(data[1]), float(data[2]), float(data[3])
                vertices.append((x, y, z))
            if data[0] == 'f':
                p0_id, p1_id, p2_id = data[1].split('/')[0], data[2].split('/')[0], data[3].split('/')[0]
                faces.append((p0_id, p1_id, p2_id))
    if len(vertices) == 0:
        return None
    else:
        return vertices, faces


def test_obj_model(file_path: str, prepare = None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    if prepare is None:
        prepare = lambda x: int(x * 7 + 400)

    points, faces = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for point in points:
        x, y = point[0], point[1]
        x, y = prepare(x), prepare(y)

        if not image.point_exist(x, y):
            continue
        image.set_pixel(x, y, color)
    image.show(f'{model_name} points')
    image.clear()

    for face in faces:
        if len(face) != 3:
            continue
        point_0_id, point_1_id, point_2_id = face[0], face[1], face[2]

        x0, y0, _ = points[point_0_id]
        x1, y1, _ = points[point_1_id]
        x2, y2, _ = points[point_2_id]

        x0, y0 = prepare(x0), prepare(y0)
        x1, y1 = prepare(x1), prepare(y1)
        x2, y2 = prepare(x2), prepare(y2)

        if not (image.point_exist(x0, y0) or image.point_exist(x1, y1) or image.point_exist(x2, y2)):
            continue
        line_4(x0, y0, x1, y1, image, color)
        line_4(x0, y0, x2, y2, image, color)
        line_4(x1, y1, x2, y2, image, color)

    image.show(f'{model_name} faces')
