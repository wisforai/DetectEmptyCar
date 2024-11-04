from aiohttp import web

from modules.deploy_handle import DeployHandle
import cv2

import numpy as np
import base64
from difflib import SequenceMatcher
import datetime
import sys
from modules.test_rec import TextRecognizer
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # 输出到终端
                        logging.FileHandler('output.txt')  # 输出到文件
                    ])

logging.info("Initializing...")

car_deploy_handle = DeployHandle(
    lib_path="./cmake-build-release/AisDeployC.dll",
    model_path="./models/" + "car.aism",
    gpu_id=0,
    language="Chinese"
)
car_deploy_handle.model_init()

empty_deploy_handle = DeployHandle(
    lib_path="./cmake-build-release/AisDeployC.dll",
    model_path="./models/empty.aism",
    gpu_id=0,
    language="Chinese"
)
empty_deploy_handle.model_init()

license_deploy_handle = DeployHandle(
    lib_path="./cmake-build-release/AisDeployC.dll",
    model_path="./models/license.aism",
    gpu_id=0,
    language="Chinese"
)
license_deploy_handle.model_init()
rec_ocr = TextRecognizer()
logging.info("Init success")

log_file = open('output.txt', 'w')

# 保存原始标准输出
original_stdout = sys.stdout

# 重定向标准输出到文件
sys.stdout = log_file


def base64_to_cv2_img(image_base64: str):
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(',')[1]
    img_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def check_base64(image_base64: str):
    prefixes = ["data:image/jpeg;base64,", "data:image/png;base64,"]
    for prefix in prefixes:
        if image_base64.startswith(prefix):
            return image_base64[len(prefix):]
    return None


def rle_area_in_mask(h, w, rle):
    mask = h * w
    count =0
    for i in range(0, len(rle), 2):
        count += rle[i + 1]
    ratio = count /mask
    return ratio


def crop_image(bbox, img, height, width):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img, x1, y1


def detect_empty(crop_img, car_x1, car_y1, client_data):
    result_empty = empty_deploy_handle.deploy(crop_img)
    empty_datas = result_empty["data"]
    empty_bbox = []
    best_empty_score = 0
    if empty_datas[0] is not None:
        for empty_data in empty_datas[0]:
            empty_score = empty_data['score']
            if empty_score < 0.5:
                continue
            if empty_score > best_empty_score:
                best_empty_score = empty_score
                empty_bbox = empty_data['bbox']
                empty_rle = empty_data['mask']['RLE']
                w, h = empty_data['size'][0], empty_data['size'][1]
        # 当前车辆为空车
        if empty_bbox:
            if rle_area_in_mask(h, w, empty_rle) > 0.15:
                client_data["empty_list"] = [empty_bbox[0] + car_x1, empty_bbox[1] + car_y1, empty_bbox[2] + car_x1, empty_bbox[3] + car_y1]
                client_data["score"] = best_empty_score
                client_data["code"] = 200
            else:
                client_data["code"] = 201
        else:
            client_data["code"] = 201
    else:
        client_data["code"] = 201
    return client_data


def detect_license(cropped_img):
    result_license = license_deploy_handle.deploy(cropped_img)
    license_datas = result_license["data"]
    if license_datas[0] is not None:
        h, w = cropped_img.shape[:2]
        best_license_score = 0
        best_license_bbox = []
        for license_data in license_datas[0]:
            score = license_data['score']
            if score < 0.5:
                continue
            license_bbox = license_data['bbox']
            if score > best_license_score:
                best_license_score = score
                best_license_bbox = license_bbox
        if best_license_bbox:
            crop_license_img, _, _ = crop_image(best_license_bbox, cropped_img, h, w)
            license_number = rec_ocr.forward(crop_license_img)
            return license_number
        else:
            return None
    else:
        return None


async def algorithm(request):
    data = await request.json()
    uuid = data.get("uuid")
    license = data.get("license")
    image_base64 = data.get("image_base64", None)

    client_data = {
        "uuid": uuid,
        "license": license,
        "code": None,
        "empty_list": [],
        "score": 0,
    }
    now = datetime.datetime.now()
    logging.info("Receive success")
    if check_base64(image_base64):
        img = base64_to_cv2_img(image_base64)

        # 识别图片中的车辆
        result_car = car_deploy_handle.deploy(img)
        car_datas = result_car["data"]
        if car_datas[0] is not None:
            logging.info("Detect {} cars".format(len(car_datas[0])))
            height, width = img.shape[:2]
            # 图中仅有一辆渣土车
            if len(car_datas[0]) == 1:
                car_bbox = car_datas[0][0]['bbox']
                cropped_img, car_x1, car_y1 = crop_image(car_bbox, img, height, width)
                client_data = detect_empty(cropped_img, car_x1, car_y1, client_data)
                logging.info(client_data)
                return web.json_response(client_data)
            # 不止一辆渣土车
            else:
                # 找到当前车牌对应的车辆
                best_similarity = 0
                for car_data in car_datas[0]:
                    car_bbox = car_data['bbox']
                    cropped_img, _, _ = crop_image(car_bbox, img, height, width)
                    license_number = detect_license(cropped_img)
                    if not license_number:
                        continue
                    else:
                        logging.info(f"License of current car is {license_number}")
                        similarity = SequenceMatcher(None, license, license_number).ratio()
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_car_bbox = car_bbox
                if best_similarity == 0.0:
                    # 当前图片未识别到给定车牌的渣土车
                    client_data["code"] = 202
                    logging.info(client_data)
                    return web.json_response(client_data)
                else:
                    cropped_img, car_x1, car_y1 = crop_image(best_car_bbox, img, height, width)
                    client_data = detect_empty(cropped_img, car_x1, car_y1, client_data)
                    logging.info(client_data)
                    return web.json_response(client_data)
        else:
            client_data["code"] = 202
            logging.info(client_data)
            return web.json_response(client_data)
    else:
        # 传输图片格式错误
        client_data["code"] = 203
        logging.info(client_data)
        return web.json_response(client_data)


app = web.Application(client_max_size=1024**2*8)
app.router.add_post('', algorithm)

if __name__ == '__main__':
    web.run_app(app, port=9610)
