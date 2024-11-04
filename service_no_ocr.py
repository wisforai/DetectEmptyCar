from aiohttp import web

from modules.deploy_handle import DeployHandle
import cv2

import numpy as np
import base64
from difflib import SequenceMatcher
import datetime

from modules.test_rec import TextRecognizer

print("[INFO]:Initializing...")
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

alpha_deploy_handle = DeployHandle(
    lib_path="./cmake-build-release/AisDeployC.dll",
    model_path="./models/alpha.aism",
    gpu_id=0,
    language="Chinese"
)
alpha_deploy_handle.model_init()



print("[INFO]:Init success")


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


def crop_car(car_bbox, img, height, width):
    car_x1, car_y1, car_x2, car_y2 = map(int, car_bbox)
    car_x1 = max(0, car_x1)
    car_y1 = max(0, car_y1)
    car_x2 = min(width, car_x2)
    car_y2 = min(height, car_y2)
    crop_img = img[car_y1:car_y2, car_x1:car_x2]
    return crop_img, car_x1, car_y1


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
                print("[INFO]:Is empty")
                client_data["code"] = 200
            else:
                print("[INFO]:Is not empty")
                client_data["code"] = 201
        else:
            print("[INFO]:Is not empty")
            client_data["code"] = 201
    else:
        print("[INFO]:Is not empty")
        client_data["code"] = 201
    return client_data


def detect_license(crop_img):
    result_license = license_deploy_handle.deploy(crop_img)
    license_datas = result_license["data"]
    if license_datas[0] is not None:
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
            license_x1, license_y1, license_x2, license_y2 = map(int, best_license_bbox)
            crop_license_img = crop_img[license_y1:license_y2, license_x1:license_x2]
            gray_license_img = cv2.cvtColor(crop_license_img, cv2.COLOR_BGR2GRAY)
            license_number = detect_alpha(gray_license_img)
            return license_number
        else:
            return None
    else:
        return None


def detect_alpha(crop_license_img):
    result_alpha = alpha_deploy_handle.deploy(crop_license_img)
    alpha_datas = result_alpha["data"]
    if alpha_datas[0] is not None:
        alpha_list = []
        for alpha_data in alpha_datas[0]:
            score = alpha_data['score']
            if score < 0.5:
                continue
            alpha_bbox = alpha_data['bbox']
            alpha = alpha_data['category']
            alpha_list.append([alpha, alpha_bbox])
        if alpha_list is None:
            return None
        else:
            # 按照x1进行车牌字符的排序
            sorted_alpha_list = sorted(alpha_list, key=lambda item: (item[1]))
            sorted_keys = [item[0] for item in sorted_alpha_list]
            license_number = ''.join(sorted_keys)
            return license_number
    else:
        return None


async def algorithm(request):
    data = await request.json()
    uuid = data.get("uuid")
    license = data.get("license")
    image_base64 = data.get("image_base64", None)  # 接收base64编码的图片

    client_data = {
        "uuid": uuid,
        "license": license,
        "code": None,
        "empty_list": [],
        "score": 0,
    }
    now = datetime.datetime.now()
    print("[INFO]:Receive success", f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}")
    if check_base64(image_base64):
        img = base64_to_cv2_img(image_base64)

        # 识别图片中的车辆
        result_car = car_deploy_handle.deploy(img)
        car_datas = result_car["data"]

        if car_datas[0] is not None:
            height, width = img.shape[:2]
            # 图中仅有一辆渣土车
            if len(car_datas[0]) == 1:
                car_bbox = car_datas[0][0]['bbox']
                crop_img, car_x1, car_y1 = crop_car(car_bbox, img, height, width)
                client_data = detect_empty(crop_img, car_x1, car_y1, client_data)
                return web.json_response(client_data)
            # 不止一辆渣土车
            else:
                # 找到当前车牌对应的车辆
                best_similarity = 0
                for car_data in car_datas[0]:
                    car_bbox = car_data['bbox']
                    crop_img, _, _ = crop_car(car_bbox, img, height, width)
                    license_number = detect_license(crop_img)
                    if not license_number:
                        continue
                    else:
                        similarity = SequenceMatcher(None, license, license_number).ratio()
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_car_bbox = car_bbox
                if best_similarity == 0.0:
                    # 当前图片未识别到给定车牌的渣土车
                    client_data["code"] = 202
                    print("[INFO]:No license plate specified")
                    return web.json_response(client_data)
                else:
                    crop_img, car_x1, car_y1 = crop_car(best_car_bbox, img, height, width)
                    client_data = detect_empty(crop_img, car_x1, car_y1, client_data)
                    return web.json_response(client_data)
        else:
            client_data["code"] = 202
            print("[INFO]:No license plate specified")
            return web.json_response(client_data)
    else:
        # 传输图片格式错误
        client_data["code"] = 203
        print("[INFO]:Image format error")
        return web.json_response(client_data)


app = web.Application(client_max_size=1024**2*4)
app.router.add_post('', algorithm)

if __name__ == '__main__':
    web.run_app(app, port=9610)
