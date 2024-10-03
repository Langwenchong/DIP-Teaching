import gradio as gr
import cv2
import numpy as np
from PIL import Image
import math

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication


def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# 获取图像中心位置以及平移逆平移矩阵


def get_center_info(image):
    center_x = image.shape[1] / 2
    center_y = image.shape[0] / 2
    # 为了以中心放缩，首先将图像原点移动到中心
    translate_matrix = to_3x3(np.array([[1, 0, -center_x], [0, 1, -center_y]]))
    # 将图像原点移回原来位置
    inv_translate_matrix = to_3x3(
        np.array([[1, 0, center_x], [0, 1, center_y]]))
    return center_x, center_y, translate_matrix, inv_translate_matrix

# Function to apply transformations based on user inputs


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3),
                         dtype=np.uint8) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[pad_size:pad_size+image.shape[0],
              pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)
    # FILL: Apply Composition Transform
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    if scale is not None:
        # 获取图像中心位置以及平移逆平移矩阵
        center_x, center_y, translate_matrix, inv_translate_matrix = get_center_info(
            transformed_image)
        # 放缩矩阵
        scale_matrix = to_3x3(np.array([[scale, 0, 0], [0, scale, 0]]))
        # 合并到一起
        transform_matrix = np.dot(
            np.dot(inv_translate_matrix, scale_matrix), translate_matrix)
        transformed_image = cv2.warpPerspective(
            transformed_image, transform_matrix, (transformed_image.shape[1], transformed_image.shape[0]))
    if rotation is not None:
        center_x, center_y, translate_matrix, inv_translate_matrix = get_center_info(
            transformed_image)
        # 角度转弧度
        theta = math.radians(rotation)
        rotation_matrix = to_3x3(np.array(
            [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0]]))
        transform_matrix = np.dot(
            np.dot(inv_translate_matrix, rotation_matrix), translate_matrix)
        transformed_image = cv2.warpPerspective(
            transformed_image, transform_matrix, (transformed_image.shape[1], transformed_image.shape[0]))
    if translation_x is not None:
        transform_matrix = np.array(
            [[1, 0, translation_x], [0, 1, 0]]).astype(np.float32)
        transformed_image = cv2.warpAffine(
            transformed_image, transform_matrix, (transformed_image.shape[1], transformed_image.shape[0]))
    if translation_y is not None:
        transform_matrix = np.array(
            [[1, 0, 0], [0, 1, translation_y]]).astype(np.float32)
        transformed_image = cv2.warpAffine(
            transformed_image, transform_matrix, (transformed_image.shape[1], transformed_image.shape[0]))
    if flip_horizontal:
        center_x, center_y, translate_matrix, inv_translate_matrix = get_center_info(
            transformed_image)
        flip_matrix = to_3x3(np.array([[-1, 0, 0], [0, 1, 0]]))
        transform_matrix = np.dot(
            np.dot(inv_translate_matrix, flip_matrix), translate_matrix)
        transformed_image = cv2.warpPerspective(
            transformed_image, transform_matrix, (transformed_image.shape[1], transformed_image.shape[0]))
        # transformed_image = cv2.flip(transformed_image, 1)
        # 转换为PIL Image格式
    transformed_image = Image.fromarray(transformed_image)
    return transformed_image

# Gradio Interface


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0,
                                  step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(
                    minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


# Launch the Gradio interface
interactive_transform().launch()
