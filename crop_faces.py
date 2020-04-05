# created at 2018-01-22
# updated at 2019-07-26

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_cut

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import os
import itertools
import uuid

def distance(pt1, pt2):
    """Returns the euclidian distance in 2D between 2 pts."""
    distance = np.linalg.norm(pt2 - pt1)
    return distance


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def bgr_to_rbg(img):
    """Given a BGR (cv2) numpy array, returns a RBG (standard) array."""
    dimensions = len(img.shape)
    if dimensions == 2:
        return img
    return img[..., ::-1]


def intersect(v1, v2):
    a1, a2 = v1
    b1, b2 = v2
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db).astype(float)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1

face_percent = 60

height = 256

width=256


def _determine_safe_zoom(imgh, imgw, x, y, w, h):
    """Determines the safest zoom level with which to add margins
    around the detected face. Tries to honor `self.face_percent`
    when possible.
    Parameters:
    -----------
    imgh: int
        Height (px) of the image to be cropped
    imgw: int
        Width (px) of the image to be cropped
    x: int
        Leftmost coordinates of the detected face
    y: int
        Bottom-most coordinates of the detected face
    w: int
        Width of the detected face
    h: int
        Height of the detected face
    Diagram:
    --------
    i / j := zoom / 100
              +
    h1        |         h2
    +---------|---------+
    |      MAR|GIN      |
    |         (x+w, y+h)|
    |   +-----|-----+   |
    |   |   FA|CE   |   |
    |   |     |     |   |
    |   ├──i──┤     |   |
    |   |  cen|ter  |   |
    |   |     |     |   |
    |   +-----|-----+   |
    |   (x, y)|         |
    |         |         |
    +---------|---------+
    ├────j────┤
              +
    """
    # Find out what zoom factor to use given self.aspect_ratio
    corners = itertools.product((x, x + w), (y, y + h))
    center = np.array([x + int(w / 2), y + int(h / 2)])
    i = np.array(
        [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
    )  # image_corners
    image_sides = [(i[n], i[n + 1]) for n in range(4)]

    corner_ratios = [face_percent]  # Hopefully we use this one
    for c in corners:
        corner_vector = np.array([center, c])
        a = distance(*corner_vector)
        intersects = list(intersect(corner_vector, side) for side in image_sides)
        for pt in intersects:
            if (pt >= 0).all() and (pt <= i[2]).all():  # if intersect within image
                dist_to_pt = distance(center, pt)
                corner_ratios.append(100 * a / dist_to_pt)
    return max(corner_ratios)


def _crop_positions(imgh, imgw, x, y, w, h,):
    """Retuns the coordinates of the crop position centered
    around the detected face with extra margins. Tries to
    honor `self.face_percent` if possible, else uses the
    largest margins that comply with required aspect ratio
    given by `self.height` and `self.width`.
    Parameters:
    -----------
    imgh: int
        Height (px) of the image to be cropped
    imgw: int
        Width (px) of the image to be cropped
    x: int
        Leftmost coordinates of the detected face
    y: int
        Bottom-most coordinates of the detected face
    w: int
        Width of the detected face
    h: int
        Height of the detected face
    """
    zoom = _determine_safe_zoom(imgh, imgw, x, y, w, h)

    if zoom > 100:
        zoom = face_percent
    # Adjust output height based on percent
    if height >= width:
        height_crop = h * 100.0 / zoom
        width_crop = (width / height) * float(height_crop)
    else:
        width_crop = w * 100.0 / zoom
        height_crop = float(width_crop) / (width / height)

    # Calculate padding by centering face
    xpad = (width_crop - w) / 2
    ypad = (height_crop - h) / 2

    # Calc. positions of crop
    h1 = x - xpad
    h2 = x + w + xpad
    v1 = y - ypad
    v2 = y + h + ypad

    return [int(v1), int(v2), int(h1), int(h2)]

# Dlib 预测器

def crop_faces(img):
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    folder_name = uuid.uuid1()
    os.mkdir(f"images/{str(folder_name)}")
    for num, face in enumerate(faces):
        height = face.bottom()-face.top()
        width = face.right()-face.left()

        pos = _crop_positions(img.shape[0], img.shape[1], face.left(), face.top(), width, height)

        image = img[pos[0]: pos[1], pos[2]: pos[3]]

        # Resize
        image = cv2.resize(
            image, (256, 256), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(f"images/{folder_name}/face_{num}.jpg", bgr_to_rbg(image))
    return len(faces), folder_name

def highlight_faces(img):
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    filename = f"images/{uuid.uuid1()}.jpg"
    for num, face in enumerate(faces):
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255,0,0), 2)

        cv2.putText(img, str(num + 1), (face.left() + (face.right() - face.left())// 3, face.bottom() + (face.top() - face.bottom()) // 3), color=(255,0,0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=3)

    cv2.imwrite(filename, bgr_to_rbg(img))
    return filename
if __name__ == "__main__":
    highlight_faces(cv2.imread("test.jpg"))