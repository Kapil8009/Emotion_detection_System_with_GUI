# train_model.py
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------
# CONFIG - edit if needed
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # emotion_detection/
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")                  # archive/ (use your dataset)
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
TEST_DIR  = os.path.join(ARCHIVE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 25

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Data generators
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Found classes:", train_gen.class_indices)

# save class indices (so capture script can map numeric predictions -> labels)
with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)

# -----------------------
# Model architecture
# -----------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# Callbacks
# -----------------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------
# Train
# -----------------------
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

print(f"Training finished. Best model saved to: {MODEL_PATH}")
