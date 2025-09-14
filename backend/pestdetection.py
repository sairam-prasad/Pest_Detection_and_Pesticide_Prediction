import os
from PIL import Image
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras.applications import MobileNetV2
from keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import CategoricalCrossentropy

#Pest Detection
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

dataset_directory = ''

#removes corupted images in the directory
def verify_and_remove_corrupted_images(directory):

  for root, _, files in os.walk(directory):
    for file in files:
      file_path = os.path.join(root,file)

      try:
        img = Image.open(file_path)
        img = img.convert('RGB')
        img.save(file_path)

      except Exception as e:
        print(f"Corrupted image found and removed {file_path}")
        os.remove(file_path)
verify_and_remove_corrupted_images(dataset_directory)

class SafeImageDataGenerator(ImageDataGenerator):

  def flow_from_directory(self, directory, target_size= (224,224), **kwargs):
    batches = super().flow_from_directory(directory, target_size= target_size, **kwargs)

    while(True):
      batch_x, batch_y = next(batches)
      safe_batch_x, safe_batch_y = [],[]

      for i, x in enumerate(batch_x):
        try:

          safe_batch_x.append(x)
          safe_batch_y.append(batch_y[i])

        except Exception as e:
          print("Skipping image due to error:",e)

      yield np.array(safe_batch_x), np.array(safe_batch_y)

image_height, image_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 40,
    width_shift_range = 0.3,
    height_shift_range= 0.3,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip= True,
    brightness_range=(0.7, 1.3),
    validation_split= 0.3
)

train_generator = train_datagen.flow_from_directory(
    dataset_directory,
    target_size = (image_height, image_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training'

)

validation_generator = train_datagen.flow_from_directory(
    dataset_directory,
    target_size = (image_height, image_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation'
)

print(train_generator.class_indices)
print(validation_generator.class_indices)

#using MobileNetV2 as a CNN model with is pretrained on imagenet dataset
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#making the initial training model to False
base_model.trainable = False  

#creating all layers using Sequential
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.05)),
    Dropout(0.5),
    Dense(22, activation='softmax')
])

initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
base_model.trainable = True
loss = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer = Adam(learning_rate=1e-5),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

epochs = 50

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('./best_pest_detection_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
        train_generator,
        validation_data= validation_generator,
        epochs = epochs,
        callbacks = callbacks
    )

train_loss, train_accuracy = model.evaluate(train_generator, verbose=1)
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)


print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
