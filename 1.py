import cv2
import numpy as np

def scale_object(image_path, mask_path, scale=1.5, output_path="output.png"):
    # 读取图像
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    assert img is not None, "Image not found"
    assert mask is not None, "Mask not found"

    h, w = img.shape[:2]

    # 二值化 mask
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 找到 mask 的 bounding box
    ys, xs = np.where(mask_bin > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    obj = img[y1:y2+1, x1:x2+1]
    obj_mask = mask_bin[y1:y2+1, x1:x2+1]

    # 原始尺寸
    oh, ow = obj.shape[:2]

    # 放大后的尺寸
    nh, nw = int(oh * scale), int(ow * scale)

    # resize 物体和 mask
    obj_scaled = cv2.resize(obj, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(obj_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # 放回位置（以中心对齐）
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2

    ny1 = max(0, cy - nh // 2)
    nx1 = max(0, cx - nw // 2)
    ny2 = min(h, ny1 + nh)
    nx2 = min(w, nx1 + nw)

    sy1 = 0
    sx1 = 0
    sy2 = ny2 - ny1
    sx2 = nx2 - nx1

    # 拷贝原图
    result = img.copy()

    # mask 融合
    region = result[ny1:ny2, nx1:nx2]
    m = mask_scaled[sy1:sy2, sx1:sx2] > 0

    region[m] = obj_scaled[sy1:sy2, sx1:sx2][m]
    result[ny1:ny2, nx1:nx2] = region

    cv2.imwrite(output_path, result)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    scale_object(
        image_path="/mnt/data-2/bob/spacelayer/data/DAVIS/JPEGImages/480p/car-turn/00030.jpg",
        mask_path="/mnt/data-2/bob/spacelayer/data/DAVIS/Annotations/480p/car-turn/00030.png",
        scale=1.4,
        output_path="/mnt/data-2/bob/spacelayer/results/scaled.png"
    )

