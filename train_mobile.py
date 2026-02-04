import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2 # <-- We use MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data/images' # This folder should contain 'cancer', 'benign', 'other'
MODEL_SAVE_PATH = 'saved_models/mobile_model.h5'
os.makedirs('saved_models', exist_ok=True)
# ---------------------

def build_mobile_model(input_shape, num_classes):
    # Load MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add our custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    
    # --- THIS IS THE KEY CHANGE ---
    # Output 3 classes (0=benign, 1=cancer, 2=other)
    # Use 'softmax' for multi-class classification
    output_layer = Dense(num_classes, activation='softmax')(x) 
    
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def main():
    # 1. Set up Data Generators
    # Use preprocessing and augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% of data for validation
    )

    print("Loading TRAINING data...")
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse', # <-- Use 'sparse' for multi-class
        subset='training'
    )

    print("Loading VALIDATION data...")
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    # 2. Build the model
    num_classes = train_generator.num_classes
    if num_classes != 3:
        print(f"ERROR: Found {num_classes} classes. Expected 3 (benign, cancer, other).")
        print(f"Please check your folders in {DATA_DIR}")
        return
        
    print("Found 3 classes. Building model...")
    model = build_mobile_model(input_shape=(*IMAGE_SIZE, 3), num_classes=3)
    
    # 3. Compile the model
    # --- THIS IS THE KEY CHANGE ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy', # <-- Loss for multi-class
        metrics=['accuracy']
    )
    
    model.summary()

    # 4. Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=10, # Start with 10, increase if needed
        validation_data=validation_generator
    )
    
    # 5. Save the final model
    model.save(MODEL_SAVE_PATH)
    print(f"\nTraining complete. Mobile model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()