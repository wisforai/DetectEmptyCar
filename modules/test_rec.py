import cv2
import numpy as np
from math import ceil
from typing import Union, Tuple, Iterator, Optional, List
from .helpers import read_image, BaseVisionTask


class TextRecognizer(BaseVisionTask):
    def __init__(self, model_path: str = "models/ch_ppocrv4_rec.onnx",
                 text_path: str = 'models/rec_word_dict.txt',
                 rec_threshold: float = 0.5,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        文本识别器
        :param model_path: 文字识别模型的路径
        :param text_path: 文本库的路径
        :param rec_threshold: 文字识别的置信度，存在意义不大
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        super().__init__(model_path, thread_num, use_gpu)

        self.rec_threshold = rec_threshold
        self._input_size = (3, 48, 320)
        self._input_mean = 127.5
        self._input_std = 127.5
        with open(text_path, 'r', encoding='utf8') as f:
            self._texts = f.read().replace('\n', '') + ' '

    def _preprocess(self, img: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        图像预处理，将文本框图像的高等比例固定尺寸
        :return: 返回处理后的图像
        """

        h, w = img.shape[:2]
        scale = self._input_size[1] / h
        obj_w = ceil(w * scale)

        img2 = cv2.resize(img, (obj_w, self._input_size[1]), interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC)
        return img2

    def _preprocess2(self, input_img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param input_img: 预处理后的图像
        :return: 可供模型输入的数据
        """
        input_tensor = input_img.transpose((2, 0, 1)).astype(np.float32)
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor[np.newaxis, ...]

    def _postprocess(self, each_output: np.ndarray) -> str:
        """
        后处理，判断文字识别结果
        :param each_output: 模型推理结果
        :return: 文字识别结果
        """
        text_idx_li = each_output.argmax(axis=1)
        content = ''.join([self._texts[i - 1] for idx, i in enumerate(text_idx_li) if i != 0 and not (idx > 0 and text_idx_li[idx - 1] == text_idx_li[idx])])
        return content

    def forward(self, img) -> str:
        """
        输入图像得到文字识别结果
        :param img_obj: 图片对象
        :return: 文字识别结果
        """
        input_img = self._preprocess(img)
        input_tensor = self._preprocess2(input_img)
        outputs = self.model.run(None, {self.input_name: input_tensor})[0][0]
        return self._postprocess(outputs)


# img = cv2.imread('../license.jpg')
# rec_orc = TextRecognizer()
#
# content = rec_orc.forward(img)
# print(content)
