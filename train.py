import tensorflow as tf
from tensorflow import keras
import os

print("TensorFlow Version:", tf.__version__)

# 1. Locate the Data
data_dir = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'

# 2. Load the Images
print("Loading dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # Use 20% of images for testing
    subset="training",
    seed=123,
    image_size=(224, 224), 
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# 3. Build the Model (Using MobileNetV2, just like Teachable Machine)
# We freeze the base model so we only train the new ASL part
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),include_top=False,weights='imagenet')
base_model.trainable = False 

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1./127.5, offset=-1), # Normalize pixels between -1 and 1
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# 4. Train the Model
print("Starting training...")
# 5 epochs is usually enough to get 90%+ accuracy on this dataset
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# 5. Save the Model
model.save('my_asl_model.h5')
print("Model saved successfully as my_asl_model.h5!")