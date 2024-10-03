import cv2
import numpy as np
import gradio as gr
from PIL import Image

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    # 参考了本篇blog https://blog.csdn.net/qq_14845119/article/details/123401102
    warped_image = np.array(image)
    # 将点数组转换为np同时浮点型坐标转换为图像uv索引,注意横向右为x对应索引0，竖向下为y对应索引1,先交换一下顺序
    # 在Numpy中更符合人操作的顺序先行再列
    p = np.ascontiguousarray(source_pts[:,[1,0]].astype(np.int32))
    q = np.ascontiguousarray(target_pts[:,[1,0]].astype(np.int32))
    # 为了实现查询需要找到输出图片->源图片的变换，所以需要交换一下计算dest->source的查询变换矩阵
    p, q = q, p
    ### FILL: 基于MLS or RBF 实现 image warping
    h,w = warped_image.shape[:2]
    # 生成网格点
    vy, vx = np.meshgrid(np.arange(w), np.arange(h))
    row = vx.shape[0]
    col = vx.shape[1]
    points_num = source_pts.shape[0]

    # 后面会用到
    reshaped_p = p.reshape(points_num, 2, 1, 1)                                              # [points_num, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, row, col), vy.reshape(1, row, col)))      # [2, row, col]

    # 计算每一个控制对非控制点的影响权重，使用坐标距离计算
    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [points_num, row, col]
    w /= np.sum(w, axis=0, keepdims=True)                                               # [points_num, row, col]
    
    pstar = np.zeros((2, row, col), np.float32)
    for i in range(points_num):
        pstar += w[i] * reshaped_p[i]                                                   # [2, row, col]

    phat = reshaped_p - pstar                                                           # [points_num, 2, row, col]
    phat = phat.reshape(points_num, 2, 1, row, col)                                        # [points_num, 2, 1, row, col]
    phat_T = phat.reshape(points_num, 1, 2, row, col)                                       # [points_num, 1, 2, row, col]
    reshaped_w = w.reshape(points_num, 1, 1, row, col)                                     # [points_num, 1, 1, row, col]
    pTwp = np.zeros((2, 2, row, col), np.float32)
    for i in range(points_num):
        pTwp += phat[i] * reshaped_w[i] * phat_T[i]
    del phat_T

    # 这里发现有的情况逆矩阵求不了因为行列式为0
    try:
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))                            # [row, col, 2, 2]
        flag = False                
    except np.linalg.linalg.LinAlgError:                
        flag = True             
        # 此时需要求伪逆矩阵，其实就是手动将行列式为0的设置为无穷大，然后求伴随矩阵A*,用A*和行列式求解
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))                                 # [row, col]
        det[det < 1e-8] = np.inf                
        reshaped_det = det.reshape(1, 1, row, col)                                    # [1, 1, row, col]
        # 这里行列式有问题，需要改正确
        adjoint = pTwp[[[1, 1], [0, 1]], [[1, 0], [0, 0]], :, :]                        # [2, 2, row, col]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, row, col]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [row, col, 2, 2]
    
    mul_left = reshaped_v - pstar                                                       # [2, row, col]
    reshaped_mul_left = mul_left.reshape(1, 2, row, col).transpose(2, 3, 0, 1)        # [row, col, 1, 2]
    mul_right = np.multiply(reshaped_w, phat, out=phat)                                 # [points_num, 2, 1, row, col]
    reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)                             # [points_num, row, col, 2, 1]
    out_A = mul_right.reshape(2, points_num, row, col, 1, 1)[0]                            # [points_num, row, col, 1, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right, out=out_A)    # [points_num, row, col, 1, 1]
    A = A.reshape(points_num, 1, row, col)                                                 # [points_num, 1, row, col]

    # 计算q_star
    reshaped_q = q.reshape((points_num, 2, 1, 1))                                            # [points_num, 2, 1, 1]
    qstar = np.zeros((2, row, col), np.float32)
    for i in range(points_num):
        qstar += w[i] * reshaped_q[i]                                                   # [2, row, col]

    transform = np.zeros((2, row, col), np.float32)
    for i in range(points_num):
        transform += A[i] * (reshaped_q[i] - qstar)
    transform += qstar

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        # 对于那些对应行列式为0的位置手动赋值此时的A为1
        transform[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transform[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transform=transform.astype(np.int16)
    transform[transform < 0] = 0
    transform[0][transform[0] > row - 1] = row-1
    transform[1][transform[1] > col - 1] = col-1

    # 重采样
    out_image = np.ones_like(warped_image)
    out_image[vx,vy] = warped_image[tuple(transform)]
    # out_image = Image.fromarray(out_image)

    return out_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
