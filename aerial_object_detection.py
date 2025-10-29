import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image
import cv2

# === TENSORFLOW (Pylance will ignore types) ===
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === YOLO ===
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(filename='aerial.log', level=logging.INFO)

def validate_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    required = [
        'classification_dataset/TRAIN/bird', 'classification_dataset/TRAIN/drone',
        'classification_dataset/VALID/bird', 'classification_dataset/VALID/drone',
        'classification_dataset/TEST/bird', 'classification_dataset/TEST/drone',
    ]
    missing = [d for d in required if not os.path.exists(os.path.join(base_dir, d))]
    if missing:
        st.error(f"Missing folders: {missing}")
        raise FileNotFoundError

def setup_data_generators():
    validate_dataset()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'classification_dataset', 'TRAIN')
    val_dir = os.path.join(base_dir, 'classification_dataset', 'VALID')
    test_dir = os.path.join(base_dir, 'classification_dataset', 'TEST')

    train_datagen = image.ImageDataGenerator(
        rescale=1.0/255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )
    val_datagen = image.ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    test_gen = val_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

    return train_gen, val_gen, test_gen, train_gen.samples, val_gen.samples, test_gen.samples

def build_custom_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
    return model

def build_transfer_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
    return model

def compile_model(model):
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, name: str, train_gen, val_gen, train_samples: int, val_samples: int):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'best_{name}.keras', save_best_only=True)
    ]
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_samples // 32),
        epochs=10,
        validation_data=val_gen,
        validation_steps=max(1, val_samples // 32),
        callbacks=callbacks
    )
    model.save(f'{name}.keras')
    return history

def evaluate_model(model, test_gen, name: str):
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    print(f"\n{name} Report:\n", classification_report(y_true, y_pred, target_names=['bird', 'drone']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'cm_{name.lower()}.png')
    plt.close()

def train_yolo():
    if YOLO is None:
        return
    yaml_content = """path: .
train: object_detection_dataset/train/images
val: object_detection_dataset/val/images
names:
  0: bird
  1: drone
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=5, imgsz=416, batch=8)

def streamlit_app():
    st.set_page_config(page_title="Aerial Detection", layout="wide")
    st.title("Bird vs Drone Detection")

    if not os.path.exists('transfer.keras'):
        st.error("Run: `python aerial_object_detection.py --train`")
        return

    model = models.load_model('transfer.keras')
    yolo_model = YOLO('runs/detect/train/weights/best.pt') if YOLO and os.path.exists('runs/detect/train/weights/best.pt') else None

    uploaded = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Input")

        # Classification
        img_array = np.array(img.resize((224, 224))) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_batch)[0]
        label = ['Bird', 'Drone'][np.argmax(pred)]
        conf = pred.max() * 100
        st.success(f"**{label}** ({conf:.1f}%)")

        # YOLO
        if yolo_model:
            results = yolo_model(np.array(img))
            result_img = results[0].plot()
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Detections")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if args.train:
        train_gen, val_gen, test_gen, t1, t2, t3 = setup_data_generators()
        custom = compile_model(build_custom_cnn())
        transfer = compile_model(build_transfer_model())

        train_model(custom, 'custom', train_gen, val_gen, t1, t2)
        train_model(transfer, 'transfer', train_gen, val_gen, t1, t2)

        evaluate_model(custom, test_gen, 'Custom')
        evaluate_model(transfer, test_gen, 'Transfer')

        train_yolo()
    else:
        streamlit_app()

if __name__ == '__main__':
    main()
    
