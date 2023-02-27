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
            self.pixels -= 1
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

    @staticmethod
    def fox_prepare(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return [int(coord * 7 + 500) for coord in x], [int(-coord * 7 + 700) for coord in y]
        return int(x * 7 + 500), int(-y * 7 + 700)

    @staticmethod
    def deer_prepare(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return [int(coord * 0.4 + 500) for coord in x], [int(-coord * 0.4 + 700) for coord in y]
        return int(x * 0.4 + 500), int(-y * 0.4 + 700)

    @staticmethod
    def rabbit_prepare(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return [int(coord * 4500 + 500) for coord in x], [int(-coord * 4500 + 700) for coord in y]
        return int(x * 4500 + 500), int(-y * 4500 + 700)

    def get_bicentric_coordinates(self, x: int, y: int, x0: float, y0: float,
                                  x1: float, y1: float, x2: float, y2: float):
        try:
            lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
            lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
            lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
            # if lambda0 + lambda1 + lambda2 == 1.0:
            return lambda0, lambda1, lambda2
        except ZeroDivisionError as zero_error:
            return 0, 0, 0
        # else:
        #     print("lambda0 + lambda1 + lambda2 != 1.0")
        #     exit(1)

    def get_triangle_bounding_box(self, x0, y0, x1, y1, x2, y2):
        xmin = min(x0, x1, x2)
        ymin = min(y0, y1, y2)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > self.width: xmax = self.width
        if ymax > self.height: ymax = self.height
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def fill_triangle(self, triangle: List, color: Tuple[int, int, int] = None):
        x0, y0, x1, y1, x2, y2 = triangle
        if color is None:
            color = (255, 255, 255)
        xmin, ymin, xmax, ymax = self.get_triangle_bounding_box(x0, y0, x1, y1, x2, y2)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                l1, l2, l3 = self.get_bicentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                if l1 > 0 and l2 > 0 and l3 > 0:
                    self.set_pixel(x, y, color)



    def fill_triangle_z(self, triangle: List, color: Tuple[int, int, int] = None):
        x0, y0, z0,  x1, y1, z1, x2, y2, z2 = triangle
        h, w = 1000, 1000
        z_buf = np.zeros((h, w), dtype= np.int8)
        z_buf += 1000
        if color is None:
            color = (255, 255, 255)
        xmin, ymin, xmax, ymax = self.get_triangle_bounding_box(x0, y0, x1, y1, x2, y2)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                l1, l2, l3 = self.get_bicentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                if l1 > 0 and l2 > 0 and l3 > 0:
                    z_ = l1*z0+l2*z1+l3*z2
                    if (z_buf[x,y] < z_):
                        self.set_pixel(x, y, color)
                        z_buf[x, y] = z_



def get_normal_vector(triangle: List):
    x0, y0, z0, x1, y1, z1, x2, y2, z2 = triangle

    try:
        v1 = np.array([x1 - x0, y1 - y0, z1 - z0])
        v2 = np.array([x1 - x2, y1 - y2, z1 - z2])
        norma = np.cross(v1, v2)
        #dlina = np.linalg.norm(normal, 2)
        return (norma)
    except ZeroDivisionError as zero_error:
        return 0, 0, 0

def drow_normal_triangle(norma):
    l = np.array([ 0, 0, 1])
    cos_like = np.dot(norma, l)
    #cos_like = np.cross(norma, l)
    dlina = np.linalg.norm(norma, 2)*np.linalg.norm(l, 2)
    return ((cos_like/ dlina))



def test_normal():
    # img = MyImage(12, 12)

    triangle_1 = [2.5, 1.5, 2.7, 5.7, 10.4, 5.6, 10.3, 5.5, 7.0]
    # triangle_2 = [0, 0, 11, 0, 13, 13]

    print(get_normal_vector(triangle_1))


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
        d_error = 0.0
    error = 0.0
    y = y0

    for x in range(x0, x1 + 1):
        img.set_pixel(y, x, color) if step else img.set_pixel(x, y, color)
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


def test_obj_model(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    points, faces = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for point in points:
        x, y = point[0], point[1]
        x, y = prepare(x, y)

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

        x0, y0 = prepare(x0, y0)
        x1, y1 = prepare(x1, y1)
        x2, y2 = prepare(x2, y2)

        if not (image.point_exist(x0, y0) or image.point_exist(x1, y1) or image.point_exist(x2, y2)):
            continue
        line_4(x0, y0, x1, y1, image, color)
        line_4(x0, y0, x2, y2, image, color)
        line_4(x1, y1, x2, y2, image, color)

    image.show(f'{model_name} faces')


def test_fill_triangle():
    img = MyImage(12, 12)

    triangle_1 = [2.5, 1.5, 5.7, 10.4, 10.3, 5.5]
    triangle_2 = [0, 0, 11, 0, 13, 13]

    img.fill_triangle(triangle_1)
    img.show('Triangle #1')
    img.clear()

    img.fill_triangle(triangle_2)
    img.show('Triangle #2')
    img.clear()


def test_fill_obj_model(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    points, faces = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1

    for face in faces:
        if len(face) != 3:
            continue
        point_0_id, point_1_id, point_2_id = face[0], face[1], face[2]

        x0, y0, _ = points[point_0_id]
        x1, y1, _ = points[point_1_id]
        x2, y2, _ = points[point_2_id]

        x0, y0 = prepare(x0, y0)
        x1, y1 = prepare(x1, y1)
        x2, y2 = prepare(x2, y2)

        color = (np.random.randint(0, 255, dtype=np.uint8),
                 np.random.randint(0, 255, dtype=np.uint8),
                 np.random.randint(0, 255, dtype=np.uint8))
        triangle = [x0, y0, x1, y1, x2, y2]

        image.fill_triangle(triangle, color)
    image.show(f'{model_name} faces')


def triangle_3d(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    points, faces = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1
    triangle = []
    for face in faces:
        if len(face) != 3:
            continue
        point_0_id, point_1_id, point_2_id = face[0], face[1], face[2]

        x0, y0, z0 = points[point_0_id]
        x1, y1, z1 = points[point_1_id]
        x2, y2, z2 = points[point_2_id]
        triangle_3d = [x0, y0, z0, x1, y1, z1, x2, y2, z2]
        cos_like = drow_normal_triangle(get_normal_vector(triangle_3d))
        if (cos_like < 0):
            x0, y0 = prepare(x0, y0)
            x1, y1 = prepare(x1, y1)
            x2, y2 = prepare(x2, y2)

            color = (255* -1 *cos_like, 0, 0)
            triangle = [x0, y0, x1, y1, x2, y2]
            image.fill_triangle(triangle, color)
    image.show(f'{model_name} faces')

def triangle_3d_with_z(file_path: str, prepare=None, model_name: str = 'Obj model'):
    h, w = 1000, 1000
    image = MyImage(h, w)
    color = (255, 255, 255)

    points, faces = read_obj(file_path)
    faces = np.array(faces, dtype=int)
    faces -= 1
    triangle = []
    for face in faces:
        if len(face) != 3:
            continue
        point_0_id, point_1_id, point_2_id = face[0], face[1], face[2]

        x0, y0, z0 = points[point_0_id]
        x1, y1, z1 = points[point_1_id]
        x2, y2, z2 = points[point_2_id]
        triangle_3d = [x0, y0, z0, x1, y1, z1, x2, y2, z2]
        cos_like = drow_normal_triangle(get_normal_vector(triangle_3d))
        if (cos_like < 0):
            x0, y0 = prepare(x0, y0)
            x1, y1 = prepare(x1, y1)
            x2, y2 = prepare(x2, y2)

            color = (255* -1 *cos_like, 0, 0)
            triangle_3d_corect = [x0, y0, z0, x1, y1, z1, x2, y2,z2]
            image.fill_triangle_z(triangle_3d_corect, color)
    image.show(f'{model_name} faces')
