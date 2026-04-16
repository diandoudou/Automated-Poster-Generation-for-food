import os
import argparse
import time
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet import MobileNet

class NimaMobileNet(tf.keras.Model):
    def __init__(self, training=True):
        super(NimaMobileNet, self).__init__()
        self.training = training
        inputs = Input(shape=(224, 224, 3))
        self.base_model = MobileNet(
            input_shape=(224, 224, 3),
            alpha=1,
            include_top=False,
            pooling='avg',
            weights=None,
            name='MobileNet_Base'
        )

        x = self.base_model(inputs)

        if self.training:
            self.dropout = Dropout(0.25, name='dropout_1')
            x = self.dropout(x)

        self.dense_relu = Dense(10, activation='relu', name='dense_relu')
        x = self.dense_relu(x)

        self.fc_last = Dense(2, name='fc_last')
        outputs = self.fc_last(x)

        self.nima_model = Model(inputs=inputs, outputs=outputs, name="NimaMobileNet")

    def call(self, x):
        return self.nima_model(x)


class FoodAesthetics:
    def __init__(self):
        super(FoodAesthetics, self).__init__()
        self.__batch_size = 1
        self.temperature = 1.536936640739441
        self.model_wrapper = NimaMobileNet(training=False)
        self.model = self.model_wrapper.nima_model
        self.model.build((self.__batch_size, 224, 224, 3))

        try:
            self.__home_path = Path(__file__).parent.resolve()
        except NameError:
            print("Warning: __file__ not defined, assuming weights are in current working directory.")
            self.__home_path = Path(os.getcwd())

        weights_path = self.__home_path / 'trained_weights.h5'
        if not weights_path.exists():
            print(f"Error: Weights file not found at {weights_path}")
            raise FileNotFoundError(f"Required weights file not found at: {weights_path}")

        # Attempt to load weights (key logic for fixing weight issues)
        print(f"Attempting to load weights from: {weights_path}")
        try:
            # Try direct loading
            self.model.load_weights(weights_path)
            print("Model weights loaded successfully.")
        except Exception as e:
            # If direct loading fails, attempt loading by name and skip mismatched layers
            print(f"Initial weight loading failed ({e}). Attempting to load by name...")
            try:
                self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print("Model weights loaded successfully (by name, with potential mismatches skipped).")
            except Exception as load_error:
                # If all attempts fail, report the error
                raise load_error

    def aesthetic_score(self, path):
        photo = np.array(self._load_image(path))
        photo = tf.image.random_crop(tf.convert_to_tensor(photo / 255.0, dtype=tf.float32), (224, 224, 3))
        logits = self.model(tf.expand_dims(photo, axis=0))
        logits_scaled = tf.math.divide(logits, self.temperature)
        score = tf.nn.softmax(logits_scaled).numpy()[:, 1].item()
        return score

    def _load_image(self, path):
        pic = Image.open(path).convert("RGB")
        width, height = pic.size
        s = max(224 / width, 224 / height)
        if width < height:
            pic_res = pic.resize((224, round(s * height)))
        else:
            pic_res = pic.resize((round(s * width), 224))
        return pic_res

    def _image2hsv(self, pic):
        img_array = np.array(self._load_image(pic))
        return cv.cvtColor(np.float32(img_array), cv.COLOR_RGB2HSV)

    def brightness(self, path):
        image = self._image2hsv(path)
        return image[:, :, 2].mean()

    def saturation(self, path):
        image = self._image2hsv(path)
        return image[:, :, 1].mean()

    def contrast(self, path):
        image = self._image2hsv(path)
        return image[:, :, 2].std()

    def clarity(self, path, thresold=0.7, scaler=255):
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum(image[:, :, 2] / scaler > thresold) / (h * w)

    def warm(self, path):
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum((image[:, :, 0] < 30) | (image[:, :, 0] > 110)) / (h * w)

    def colourfulness(self, path):
        image = np.array(self._load_image(path))
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        sigma = np.sqrt(np.square(np.std(rg)) + np.square(np.std(yb)))
        mu = np.sqrt(np.square(np.mean(rg)) + np.square(np.mean(yb)))
        c = sigma + 0.3 * mu  # colourfulness
        return c

    def _grabcut(self, path, n_iter=10):
        image = np.array(self._load_image(path))
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)
        cv.grabCut(image, mask, rect, bgdModel, fgdModel, n_iter, cv.GC_INIT_WITH_RECT)
        return np.where((mask == cv.GC_PR_BGD) | (mask == cv.GC_BGD), 0, 1).astype('uint8')

    def size_difference(self, path):
        mask = self._grabcut(path)
        h, w = mask.shape
        counts = np.unique(mask, return_counts=True)
        if len(counts[1]) < 2:
            return 0.0
        n_back = counts[1][0]
        n_for = counts[1][1]
        return (n_for - n_back) / (h * w)

    def color_difference(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)
        front_pixels = image[mask == 1].reshape(-1, 3)
        back_pixels = image[mask == 0].reshape(-1, 3)
        avg_front = front_pixels.mean(axis=0) if front_pixels.size > 0 else np.zeros(3)
        avg_back = back_pixels.mean(axis=0) if back_pixels.size > 0 else np.zeros(3)
        return np.linalg.norm(avg_front - avg_back)

    def texture_difference(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)
        front_pixels_count = np.sum(mask == 1)
        front = np.where(mask[:, :, np.newaxis] == 1, image, 0).astype(np.uint8)
        back_pixels_count = np.sum(mask == 0)
        back = np.where(mask[:, :, np.newaxis] == 0, image, 0).astype(np.uint8)
        edges_front = cv.Canny(front, 50, 150)
        edges_back = cv.Canny(back, 50, 150)

        # Edge Density
        front_density = np.sum(edges_front) / front_pixels_count if front_pixels_count > 0 else 0
        back_density = np.sum(edges_back) / back_pixels_count if back_pixels_count > 0 else 0
        return np.abs(front_density - back_density)

    def segment_colorfulness(self, path):
        image = np.array(self._load_image(path))
        mask = self._grabcut(path)
        (B, G, R) = cv.split(image.astype("float"))

        front_mask = (mask == 1)
        R_fg = R[front_mask]
        G_fg = G[front_mask]
        B_fg = B[front_mask]

        if R_fg.size == 0:
            return 0.0

        rg = np.absolute(R_fg - G_fg)
        yb = np.absolute(0.5 * (R_fg + G_fg) - B_fg)

        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
        return stdRoot + (0.3 * meanRoot)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate aesthetic and attribute scores for food images.'
    )
    parser.add_argument('--img_dir',
                        help='Path to images folder (e.g., 55)',
                        type=dir_path,
                        required=True)
    parser.add_argument('--csv_file',
                        help='Output path and filename for the CSV file (e.g., output/feature.csv)',
                        type=str,
                        required=True)
    args = parser.parse_args()

    path_dir = Path(args.img_dir)
    csv_out_path = Path(args.csv_file)

    try:
        fa = FoodAesthetics()
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please place the 'trained_weights.h5' file in the same directory as this script.")
        exit(1)

    print('Starting to score pictures...')

    # Score images
    images = os.listdir(path_dir)
    images = [image for image in images if image.lower().endswith(('.jpg', '.jpeg'))]

    results = []
    print(f"Found {len(images)} images in {path_dir.resolve()}.")

    attribute_methods = [
        "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
        "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
    ]

    total_start_time = time.time()

    for image in images:
        start_time = time.time()
        image_path = path_dir / image
        image_data = {"image": image}

        try:
            image_data["aesthetic_score"] = fa.aesthetic_score(image_path)
            for attr_name in attribute_methods:
                attr_func = getattr(fa, attr_name)
                image_data[attr_name] = attr_func(image_path)
            results.append(image_data)
            end_time = time.time()
            print(f"Scored image: {image} in {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Error evaluating {image}: {e}")
            continue
    total_end_time = time.time()
    print(f"\nScored all pictures. Total time: {total_end_time - total_start_time:.2f}s")

    if not results:
        print("No results to export. Exiting.")
        exit(0)

    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(csv_out_path, index=False)
    print(f'Exported the CSV file correctly to {csv_out_path.resolve()}')