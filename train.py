import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import preprocess_image_from_path
from model import build_hybrid_model

# --- Configuration ---
IMAGE_SIZE = (224, 224, 3)
CSV_PATH = 'data/clinical_data.csv'
MODEL_SAVE_PATH = 'saved_models/hybrid_model.h5'
SCALER_SAVE_PATH = 'saved_models/clinical_data_scaler.pkl'
CNN_BRANCH_SAVE_PATH = 'saved_models/cnn_branch_model.h5' # <-- ADDED
CLINICAL_FEATURES = ['age', 'smoking_status', 'alcohol_use'] # Must match CSV!

# Create directories if they don't exist
os.makedirs('saved_models', exist_ok=True)
# ---------------------

def load_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # 1. Extract labels
    labels = df.pop('diagnosis').values
    
    # 2. Extract image paths
    image_paths = df.pop('image_path').values
    
    # 3. Process clinical data
    clinical_data = df[CLINICAL_FEATURES].values
    
    # Scale clinical data
    scaler = MinMaxScaler()
    clinical_data_scaled = scaler.fit_transform(clinical_data)
    
    # Save the scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    # 4. Load and preprocess images
    print("Loading and preprocessing images...")
    images_data = []
    valid_labels = []
    valid_clinical_data = []
    
    for i, path in enumerate(image_paths):
        try:
            img = preprocess_image_from_path(path)
            images_data.append(img)
            valid_labels.append(labels[i])
            valid_clinical_data.append(clinical_data_scaled[i])
        except Exception as e:
            print(f"Warning: Skipping image {path}. Error: {e}")
            
    # Convert lists to numpy arrays
    images_data = np.array(images_data)
    valid_clinical_data = np.array(valid_clinical_data)
    valid_labels = np.array(valid_labels)
    
    return images_data, valid_clinical_data, valid_labels

def main():
    # Load and preprocess all data
    images_data, clinical_data, labels = load_data(CSV_PATH)
    
    if len(images_data) == 0:
        print("No images were loaded. Exiting training.")
        return

    # Split the data into training and validation sets
    print("Splitting data...")
    X_train_img, X_val_img, X_train_cli, X_val_cli, y_train, y_val = train_test_split(
        images_data, 
        clinical_data, 
        labels, 
        test_size=0.2, # 20% for validation
        random_state=42,
        stratify=labels # Ensure balanced split
    )
    
    # Build the model
    print("Building model...")
    MLP_INPUT_SHAPE = (clinical_data.shape[1],)
    
    # --- THIS IS THE KEY FIX ---
    # Receive BOTH models from the build function
    model, cnn_branch_model = build_hybrid_model(IMAGE_SIZE, MLP_INPUT_SHAPE)
    # --- END OF FIX ---
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    print("--- HYBRID MODEL ---")
    model.summary()
    
    # Set up EarlyStopping
    early_stopper = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        [X_train_img, X_train_cli], 
        y_train,
        validation_data=([X_val_img, X_val_cli], y_val),
        epochs=25, # Increase epochs, EarlyStopping will handle it
        batch_size=32,
        callbacks=[early_stopper]
    )
    
    # --- SAVE BOTH MODELS ---
    # Save the final hybrid model
    model.save(MODEL_SAVE_PATH)
    print(f"\nTraining complete. Hybrid model saved to {MODEL_SAVE_PATH}")
    
    # Save the CNN branch model
    print("Saving CNN branch model for XAI...")
    cnn_branch_model.save(CNN_BRANCH_SAVE_PATH)
    print(f"CNN branch model saved to '{CNN_BRANCH_SAVE_PATH}'")
    # --- END OF SAVING BLOCK ---
    
    # Evaluate on validation set
    print("Evaluating model on validation set...")
    loss, acc = model.evaluate([X_val_img, X_val_cli], y_val)
    print(f"Final Validation Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()