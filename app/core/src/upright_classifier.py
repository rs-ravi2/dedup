
from abc import ABC, abstractmethod
import numpy as np
# import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import cv2

class Model(ABC):

    @abstractmethod
    def load(self):
        """Loads the model"""
    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """Preprocess the input data"""
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Run inferenece on the given input data"""
    @abstractmethod
    def run(self, *args, **kwargs):
        """The main process function for the model"""


class IdUprightClassifier(Model):

    def __init__(self, UPRIGHT_MODEL_PATH, classifier_img_size, upright_target_labels):
        self.model_path = UPRIGHT_MODEL_PATH
        self.img_size = classifier_img_size  #480
        self.target_labels = upright_target_labels

        self.load(self.model_path)

    def load(self, model_path):
#         logger.debug("Loading ID Upright Classifier")
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image, img_size):
#         logger.debug("Preprocessing Image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return image_rgb

    def predict(self, image, target_labels):
#         logger.debug("Predicting on the given image")
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
#         print(f"Upright prediction {prediction}")
        predicted_class = np.argmax(prediction, axis=-1)[0]
#         print(f"Upright prediction class {predicted_class}")
#         print(f"Upright prediction label {target_labels[predicted_class]}")
        return target_labels[predicted_class]

    def run(self, image: np.ndarray) -> float:
#         logger.debug("Running the model")

        processed_image = self.preprocess(image, self.img_size)
#         print(f"Upright processed image {processed_image.shape}")
        prediction = self.predict(processed_image, self.target_labels)

        return prediction

