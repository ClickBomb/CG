from CG_lab1 import *
from CG_lab1.MyImage import *


if __name__ == "__main__":
    # lab №1
    # №1
    # test.test()

    # №2, №3
    # MyImage.test_lines()

    # №4, №5, №6, №7
    # file_path = './obj_files/rabbit.obj'
    # MyImage.test_obj_model(file_path, prepare=MyImage.MyImage.rabbit_prepare)

    # lab №2
    # MyImage.test_fill_triangle()
    file_path = './obj_files/rabbit.obj'
    test_fill_obj_model(file_path, prepare=MyImage.rabbit_prepare, model_name='deer')
    triangle_3d(file_path, prepare=MyImage.rabbit_prepare, model_name='deer')
    triangle_3d_with_z(file_path, prepare=MyImage.rabbit_prepare, model_name='deer')
    test_normal()