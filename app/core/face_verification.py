import cv2
import numpy as np
from app.core.src.face_analysis import FaceAnalysis

model = FaceAnalysis(allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])
# model.prepare(ctx_id=0)

import math

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)


class FaceVerification:

    def __init__(self):
        self.similarity = None
        self.ref_image = None
        self.target_image = None
        self.ref_face = None
        self.target_face = None

    def load_data(self, ref_image_bytes=None, target_image_bytes=None):
        """

        :param ref_image_bytes:
        :param target_image_bytes:
        :return:
        """
        if ref_image_bytes is None or target_image_bytes is None:
            raise Exception("Both ID Image and Subscriber Image should be present in the input.")
        try:
            self.ref_image = cv2.imdecode(np.fromstring(ref_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            self.target_image = cv2.imdecode(np.fromstring(target_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            raise Exception("Unable to convert bytes to image. {}".format(str(e)))

    def predict(self):
        """

        :return:
        """

        ref_pred = model.get(self.ref_image)
        target_pred = model.get(self.target_image)

        self.ref_num_faces = len(ref_pred)
        self.target_num_faces = len(target_pred)

        if self.ref_num_faces == 0 or self.target_num_faces == 0:
            return

        similarity = cosine_measure(ref_pred[0].embedding, target_pred[0].embedding)
        return similarity

    def process(self, ref_image_bytes=None, target_image_bytes=None):
        """

        :param ref_image_bytes:
        :param target_image_bytes:
        :return:
        """
        self.load_data(ref_image_bytes=ref_image_bytes, target_image_bytes=target_image_bytes)
        similarity = self.predict()
        return similarity