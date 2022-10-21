from dataset import OxfordPets
from tensorflow import keras
from unet import get_model
import random
import os

import mlflow
import mlflow.keras

# Add mlflow logging
mlflow.autolog()

input_dir = os.path.join("data","images")
target_dir = os.path.join("data","annotations","trimaps")
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
target_img_paths = sorted([
    os.path.join(target_dir, fname)
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
])

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# Set the keras callback using ModelCheckpoint to save the best results of the model
callbacks = keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model, doing validation at the end of each epoch.
# Make sure to include the calllback for mlflow
epochs = 15
with mlflow.start_run() as run:
    model.fit(train_gen, epochs=epochs, validation_data=val_gen)
model.save('./segmentation/')

print("run_id: {}".format(r.info.run_id))
print("metrics: {}".format(r.data.metrics))