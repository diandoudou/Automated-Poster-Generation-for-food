import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]


# --- Model Definition (NimaMobileNetV2 Multi-Output) ---
class NimaMobileNetV2MultiOutput(tf.keras.Model):
    def __init__(self, training=True):
        super(NimaMobileNetV2MultiOutput, self).__init__()
        self.training = training

        inputs = Input(shape=(224, 224, 3), name='input_image')

        self.base_model = MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, pooling='avg',
            weights='imagenet',
            name='MobileNetV2_Base'
        )
        x = self.base_model(inputs)

        if self.training:
            self.dropout_layer = Dropout(0.25, name='dropout_1')
            x = self.dropout_layer(x)

        shared_features = Dense(10, activation='relu', name='shared_dense_relu')(x)

        self.aesthetic_head = Dense(1, activation='sigmoid', name='aesthetic_score')(shared_features)
        self.attribute_heads = {}
        output_list = [self.aesthetic_head]

        for name in ATTRIBUTE_NAMES:
            head = Dense(1, activation='sigmoid', name=f'{name}_output')(shared_features)
            self.attribute_heads[name] = head
            output_list.append(head)

        self.nima_model = Model(inputs=inputs, outputs=output_list, name="NimaMobileNetV2_Multi")

    def call(self, x):
        return self.nima_model(x)


class GPDDataGenerator(Sequence):
    """
    Keras Sequence for loading image data and multi-output labels (aesthetic + attributes).
    """
    def __init__(self, df, img_dir, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = df
        self.img_dir = Path(img_dir)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_and_preprocess(self, img_path):
        try:
            pic = Image.open(img_path).convert("RGB")
            pic_res = pic.resize(self.target_size)
            photo = np.array(pic_res, dtype=np.float32)
            return photo / 255.0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]

        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        Y = {}
        Y['aesthetic_score'] = np.empty((len(batch_df), 1), dtype=np.float32)
        for name in ATTRIBUTE_NAMES:
            Y[f'{name}_output'] = np.empty((len(batch_df), 1), dtype=np.float32)

        for i, row in enumerate(batch_df.itertuples()):
            img_path = self.img_dir / row.image
            X[i,] = self._load_and_preprocess(img_path)
            Y['aesthetic_score'][i] = getattr(row, 'aesthetic_score')

            for name in ATTRIBUTE_NAMES:
                Y[f'{name}_output'][i] = getattr(row, name)
        return X, Y

def fine_tune_model(epochs, csv_file, img_dir, output_model_path, pretrained_model_path=None):
    print("Starting training process with TensorFlow/Keras...")

    # Data Loading and Preparation
    data_df = pd.read_csv(csv_file)
    data_df['image'] = data_df['image'].astype(str)

    required_cols = ['image', 'aesthetic_score'] + ATTRIBUTE_NAMES
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
    train_generator = GPDDataGenerator(train_df, img_dir, batch_size=32, shuffle=True)
    val_generator = GPDDataGenerator(val_df, img_dir, batch_size=32, shuffle=False)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # Model initialization
    model_wrapper = NimaMobileNetV2MultiOutput(training=True)
    model = model_wrapper.nima_model
    model.build((None, 224, 224, 3))

    if pretrained_model_path:
        print(f"Attempting to load weights from: {pretrained_model_path}")
        try:
            model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)
            print(f"Pre-trained weights loaded successfully from {pretrained_model_path}.")
        except Exception as e:
            print(f"WARNING: Failed to load pre-trained weights. {e}")

    # Loss function and optimizer setup
    loss_weights = {'aesthetic_score': 1.0}
    attribute_loss_weight = 0.01
    for name in ATTRIBUTE_NAMES:
        loss_weights[f'{name}_output'] = attribute_loss_weight

    loss_functions = {name: MeanSquaredError() for name in loss_weights.keys()}

    base_model = model.get_layer('MobileNetV2_Base')
    base_model.trainable = False

    initial_learning_rate = 0.0001
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate),
        loss=loss_functions,
        loss_weights=loss_weights,
        metrics={'aesthetic_score': 'mae'}
    )

    print("\n--- Phase A: Training Top Layers (Epochs 1-5) ---")
    model.fit(train_generator, validation_data=val_generator, epochs=5)

    base_model.trainable = True
    freeze_at = int(len(base_model.layers) * 0.7)
    for i, layer in enumerate(base_model.layers):
        if i < freeze_at:
            layer.trainable = False

    fine_tune_learning_rate = 0.00001
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_learning_rate),
        loss=loss_functions,
        loss_weights=loss_weights,
        metrics={'aesthetic_score': 'mae'}
    )

    checkpoint_filename = 'best_model.weights.h5'
    checkpoint = ModelCheckpoint(
        os.path.join(output_model_path, checkpoint_filename),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        verbose=1
    )

    print("\n--- Phase B: Fine-Tuning Entire Model ---")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        initial_epoch=5,
        callbacks=[checkpoint]
    )

    print(f"\nTraining completed. Best model weights saved to {output_model_path}/{checkpoint_filename}")

if __name__ == '__main__':
    def check_dir(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Directory not found: {path}")
        return Path(path)

    parser = argparse.ArgumentParser(description="Train aesthetic score and attribute prediction model (TensorFlow)")
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (total)')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='path to the csv file with image names and labels (aesthetic + attributes)')
    parser.add_argument('--img_dir', type=str, required=True, help='directory with all the images')
    parser.add_argument('--output_model_path', type=str, required=True,
                        help='path to save the trained model checkpoints')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='path to the pretrained model weights (.h5 file)')
    args = parser.parse_args()

    if not os.path.exists(args.output_model_path):
        os.makedirs(args.output_model_path)

    fine_tune_model(args.epochs, args.csv_file, args.img_dir, args.output_model_path, args.pretrained_model_path)