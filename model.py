from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def create_cnn_branch(input_shape, name="cnn_branch"):
    """Creates the CNN branch using transfer learning with ResNet50."""
    
    # Load ResNet50 pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model
    base_model.trainable = False
    
    # Get the output of the base model
    x = base_model.output
    
    # Add our custom layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    
    # Create the CNN branch model
    model = Model(inputs=base_model.input, outputs=x, name=name)
    return model

def create_mlp_branch(input_shape, name="mlp_branch"):
    """Creates the MLP branch for clinical data."""
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Add dense layers
    x = Dense(32, activation='relu')(input_layer)
    x = Dropout(0.3)(x) # Dropout for regularization
    x = Dense(16, activation='relu')(x)
    
    # Create the MLP branch model
    model = Model(inputs=input_layer, outputs=x, name=name)
    return model

def build_hybrid_model(cnn_input_shape, mlp_input_shape):
    """
    Builds the final hybrid model and returns BOTH the hybrid model
    and the cnn_branch model.
    """
    
    # 1. Create the two branches
    cnn_branch = create_cnn_branch(cnn_input_shape)
    mlp_branch = create_mlp_branch(mlp_input_shape)
    
    # 2. Combine the outputs of the two branches
    combined_output = concatenate([cnn_branch.output, mlp_branch.output])
    
    # 3. Add final classification layers
    x = Dense(32, activation='relu')(combined_output)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x) # Sigmoid for binary classification
    
    # 4. Create the final model
    hybrid_model = Model(
        inputs=[cnn_branch.input, mlp_branch.input], 
        outputs=x, 
        name="hybrid_oral_cancer_model"
    )
    
    # 5. Return BOTH models
    return hybrid_model, cnn_branch

if __name__ == "__main__":
    # Test building the model
    IMG_SHAPE = (224, 224, 3)
    MLP_SHAPE = (3,) # e.g., age, smoking_status, alcohol_use
    
    # Update test call to receive both models
    model, cnn_branch = build_hybrid_model(IMG_SHAPE, MLP_SHAPE)
    
    print("--- HYBRID MODEL SUMMARY ---")
    model.summary()
    
    print("\n--- CNN BRANCH SUMMARY ---")
    cnn_branch.summary()