import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Classes.Model import create_model  # Your custom model in Model.py

# Create 'runs' folder if not exists
os.makedirs('runs', exist_ok=True)

# Image settings
img_height, img_width = 224, 224
batch_size = 32
epochs = 25

# Dataset paths

train_dir = 'C:/Users/Mamat/Downloads/LungDiseaseproject/data/train'
val_dir = 'C:/Users/Mamat/Downloads/LungDiseaseproject/data/val'




# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Save class label indices to JSON
with open('Classes/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("✅ Saved class indices to Classes/class_indices.json")

# Create model using imported function
model = create_model(img_height, img_width, num_classes=len(train_generator.class_indices))

# Callbacks for early stopping and saving best model
checkpoint = ModelCheckpoint(
    filepath='runs/best_weight.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Model training
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping]
)

print("✅ Training complete.")
print("✅ Best model saved at 'runs/best_weight.h5'")
