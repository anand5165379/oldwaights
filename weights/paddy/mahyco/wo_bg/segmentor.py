import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model('paddy_pre_unet.h5', compile=False)


def image_post_process(output, image):
    # define closing kernel
    kernel = np.ones((3, 3), np.uint8)
    
    contour = (output[..., 1] * 255).astype('uint8')
    # threshold and close small holes in the contour
    ret, contour = cv2.threshold(contour, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    fg = (output[..., 2] * 255).astype('uint8')
    # threshold and close small holes in the foreground
    ret, fg = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
    # mark sure foregrounds (patches of foreground where there are confirmed seeds)
    ret, markers = cv2.connectedComponents(fg)

    # add 10 to markers for reasons unknown
    markers = markers + 10    
    # mark background to 0
    markers[contour == 255] = 0

    # do watershed on markers
    markers = cv2.watershed(image, markers)

    # mark edges, borders and background as 0
    markers[markers == -1] = 0
    markers[markers == 10] = 0
    return markers


def add_padding(img: np.ndarray, background=(0, 0, 0)) -> np.ndarray:
    """
    Add extra padding (15% of acutual height and with) to an image with
    a background color

    :param img: image to be padded
    :param background: background color to pad
    :return: Padded image
    """
    ht, wd, cc = img.shape
    vertical_padding = int(ht * 0.15)
    horizontal_padding = int(wd * 0.15)
    img = cv2.copyMakeBorder(
        img,
        vertical_padding, vertical_padding,
        horizontal_padding, horizontal_padding,
        cv2.BORDER_CONSTANT, value=background
    )
    return img


def predict_mask(np_gray):
    resized = cv2.resize(np_gray, (512, 512))
    inp = np.expand_dims(resized, axis=0) / 255.0
    return model.predict(inp).squeeze()


def get_cropped(contour, img, multiplier=1):
    padd = 109 * 2
    padd_2 = 100
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = rect[1][0] + padd
    height = rect[1][1] + padd
    all_x = [i[0] for i in box]
    all_y = [i[1] for i in box]
    padd = 109
    x1 = min(all_x) - padd - padd_2
    x2 = max(all_x) + padd + padd_2
    y1 = min(all_y) - padd - padd_2
    y2 = max(all_y) + padd + padd_2

    if max((width, height)) < 600:
        return None

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(multiplier * (x2 - x1)), int(multiplier * (y2 - y1)))
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    cropped_w = width if not rotated else height
    cropped_h = height if not rotated else width

    if cropped_h > cropped_w:
        cropped_rotated = cv2.getRectSubPix(
            cropped, (int(cropped_w * multiplier), int(cropped_h * 1)),
            (size[0] / 2, size[1] / 2)
        )
    else:
        cropped_rotated = cv2.getRectSubPix(
            cropped, (int(cropped_w * 1), int(cropped_h * multiplier)),
            (size[0] / 2, size[1] / 2)
        )

    if cropped_rotated.shape[1] > cropped_rotated.shape[0]:
        cropped_rotated = cv2.rotate(cropped_rotated, cv2.ROTATE_90_CLOCKWISE)
    return cropped_rotated


def rescale_and_pad_np_img_to(
        np_image: np.ndarray,
        target_size,
        background=(25, 55, 115)
) -> np.ndarray:
    """
    Pads the image with background color to match the target_size
    aspect ratio. And then resizes the image to the target size.

    :param np_image: Input image
    :param target_size: Size to be resized to
    :param background: Background color to be added as padding
    :return: Padded and resized image
    """
    w_t, h_t = target_size
    h, w, c = np_image.shape

    if w / h < w_t / h_t:
        new_w = h * w_t / h_t
        padding = new_w - w
        image = cv2.copyMakeBorder(
            np_image,
            0, 0,
            int(padding // 2), int(padding // 2),
            cv2.BORDER_CONSTANT, value=background
        )
    else:
        new_h = w * h_t / w_t
        padding = new_h - h
        image = cv2.copyMakeBorder(
            np_image,
            int(padding // 2), int(padding // 2),
            0, 0,
            cv2.BORDER_CONSTANT, value=background
        )
    return cv2.resize(image, target_size)


def segment_image(np_bgr):
    h, w, _ = np_bgr.shape
    extracted_images = []

    resized_bgr = cv2.resize(np_bgr, (512, 512))
    np_gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    mask = predict_mask(np_gray)
    markers = image_post_process(mask, resized_bgr.copy())

    markers = cv2.erode(markers.astype('uint8'), None, iterations=1)
    dilated_markers = cv2.dilate(markers, None, iterations=2)

    markers = add_padding(np.expand_dims(cv2.resize(markers, (w, h)), axis=-1))
    dilated_markers = add_padding(np.expand_dims(cv2.resize(dilated_markers, (w, h)), axis=-1))
    np_bgr = add_padding(cv2.resize(np_bgr, (w, h)), (55, 34, 33))

    np_bgr[dilated_markers == 0] = 0
    contour_list, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)

    for contour in contour_list:
        seed_image = get_cropped(contour, np_bgr)
        if seed_image is not None:
            extracted_images.append(rescale_and_pad_np_img_to(
                seed_image, (224, 448), (0, 0, 0)
            ))

    return extracted_images
