#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMAGE_SIZE = (100, 100)

BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'product_classifier_model.h5'
DATASET_TRAIN = "D:/aicte/Dataset/archive/fruits-360_100x100/fruits-360/Training"  # Base folder with subfolders for each class
DATASET_TEST = "D:/aicte/Dataset/archive/fruits-360_100x100/fruits-360/Test"
def create_model(num_classes):
    """Create a convolutional neural network for image classification"""
    model = Sequential([
        # First convolutional block
        #Input(shape=(100, 100, 3)),
        #Conv2D(32, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators():
    """Prepare train and validation data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_TRAIN,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    valid_generator = valid_datagen.flow_from_directory(
        DATASET_TRAIN,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, valid_generator

def train_model():
    """Train the image classification model"""
    # Check if dataset exists
    if not os.path.exists(DATASET_TRAIN):
        print(f"Error: Dataset folder '{DATASET_TRAIN}' not found.")
        print("Please create a 'dataset' folder with subfolders for each class.")
        print("Example structure:")
        print("  dataset/")
        print("    ├── apple/")
        print("    ├── banana/")
        print("    ├── laptop/")
        print("    └── shirt/")
        return
        
    # Prepare data generators
    train_generator, valid_generator = prepare_data_generators()
    
    # Get the number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}")
    
    # Create and train the model
    model = create_model(num_classes)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks
    )
    
    # Save class indices for inference
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    np.save('class_labels.npy', class_labels)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Class labels saved to class_labels.npy")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def test_inference():
    """Test inference with the trained model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
        
    if not os.path.exists('class_labels.npy'):
        print(f"Error: Class labels file 'class_labels.npy' not found.")
        return
        
    # Load the model and class labels
    model = load_model(MODEL_PATH)
    class_labels = np.load('class_labels.npy', allow_pickle=True).item()
    
    # Create a data generator for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        DATASET_TEST,
        target_size=IMAGE_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get a sample image
    for i in range(5):  # Test 5 random images
        x, y_true = next(test_generator)
        y_pred = model.predict(x)
        
        # Get predicted class
        pred_class_idx = np.argmax(y_pred)
        true_class_idx = np.argmax(y_true)
        
        pred_class_label = class_labels[pred_class_idx]
        true_class_label = class_labels[true_class_idx]
        
        # Display results
        print(f"Prediction {i+1}:")
        print(f"True class: {true_class_label}")
        print(f"Predicted class: {pred_class_label}")
        print(f"Confidence: {y_pred[0][pred_class_idx]:.4f}")
        print("-" * 30)
        
        # Display the image
        plt.figure(figsize=(4, 4))
        plt.imshow(x[0])
        plt.title(f"True: {true_class_label}\nPred: {pred_class_label}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_PATH}' already exists.")
        response = input("Do you want to retrain the model? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training. Testing inference...")
            test_inference()
            exit()
    
    # Train the model
    print("Training model...")
    train_model()
    
    # Test inference
    print("\nTesting inference...")
    test_inference()


# In[4]:


import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class ProductClassifier:
    def __init__(self, model_path='product_classifier_model.h5', labels_path='class_labels.npy'):
        """Initialize the product classifier
        
        Args:
            model_path: Path to the trained model file
            labels_path: Path to the class labels file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_labels = None
        self.image_size = (100, 100)
        
        # Load model and class labels
        self.load()
    
    def load(self):
        """Load the model and class labels"""
        # Check if files exist
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found.")
            return False
            
        if not os.path.exists(self.labels_path):
            print(f"Error: Class labels file '{self.labels_path}' not found.")
            return False
        
        try:
            # Load the model
            self.model = load_model(self.model_path)
            
            # Load class labels
            self.class_labels = np.load(self.labels_path, allow_pickle=True).item()
            
            print(f"Model loaded successfully with {len(self.class_labels)} classes.")
            return True
            
        except Exception as e:
            print(f"Error loading model or class labels: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for the model
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # If image is a file path, open it
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            # Resize image
            image = image.resize(self.image_size)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to array
            image = img_to_array(image)
        
        # Normalize pixel values
        image = image / 255.0
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def classify_image(self, image):
        """Classify an image
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Dictionary with classification results
        """
        if self.model is None or self.class_labels is None:
            return {"error": "Model not loaded. Call load() first."}
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image)[0]
            
            # Get top predictions
            top_indices = predictions.argsort()[-5:][::-1]  # Top 5 predictions
            
            results = []
            for idx in top_indices:
                label = self.class_labels[idx]
                probability = float(predictions[idx])
                results.append({
                    "tagName": label,
                    "probability": probability
                })
                
            return {
                "predictions": results
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_classes(self):
        """Get list of classes the model can recognize"""
        if self.class_labels is None:
            return {"error": "Class labels not loaded"}
            
        return list(self.class_labels.values())

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize classifier
    classifier = ProductClassifier()
    
    # Check if model is loaded
    if classifier.model is None:
        print("Please train the model first using train_image_classifier.py")
        exit()
    
    # Test with sample image if provided
    sample_path = "D:/aicte/Dataset/archive/fruits-360_100x100/fruits-360/Test"  # Replace with path to test image
    
    if os.path.exists(sample_path) and os.path.isdir(sample_path):
        # Find a random image in the dataset
        for root, dirs, files in os.walk(sample_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    break
            else:
                continue
            break
        
        # Classify the image
        print(f"Testing with image: {image_path}")
        result = classifier.classify_image(image_path)
        
        # Display results
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            # Show the image
            img = Image.open(image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            
            # Print predictions
            print("\nPredictions:")
            for i, pred in enumerate(result["predictions"]):
                print(f"{i+1}. {pred['tagName']}: {pred['probability']:.4f}")
                
            # Add predictions to plot title
            top_pred = result["predictions"][0]
            plt.title(f"Top prediction: {top_pred['tagName']} ({top_pred['probability']:.2%})")
            plt.show()
    else:
        print(f"Sample image path '{sample_path}' not found. Skipping test.")


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Define model path
MODEL_PATH = 'customer_spend_model.pkl'

# Function to load or create the dataset
def load_or_create_dataset():
    csv_path = "D:/aicte/Dataset/archive_customer/Mall_Customers.csv"
    
    # Check if dataset exists, if not create it
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found, creating sample data...")
        # Sample data
        data = {
            'customer_id': list(range(1001, 1021)),
            'age': np.random.randint(20, 70, 20),
            'income': np.random.randint(25000, 100000, 20),
            'visit_frequency': np.random.randint(1, 25, 20),
            'avg_basket_size': np.round(np.random.uniform(2, 10, 20), 1),
            'purchase_category': np.random.choice(['groceries', 'electronics', 'clothing'], 20),
            'season': np.random.choice(['winter', 'spring', 'summer', 'fall'], 20),
            'day_of_week': np.random.choice(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], 20),
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening'], 20),
            'total_spend': np.round(np.random.uniform(30, 350, 20), 2)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Created sample dataset and saved to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing dataset from {csv_path}")
    
    print(f"Dataset shape: {df.shape}")
    print("\nData overview:")
    print(df.head())
    
    return df

# Load the dataset
df = load_or_create_dataset()

# Separate features and target
X = df.drop('Spending', axis=1)
y = df['Spending']

# Identify categorical and numerical columns
categorical_cols = ['Gender']
numerical_cols = ['CustomerID','Age', 'Income', ]

# Data preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model performance:")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance analysis
feature_names = numerical_cols + [f"{col}_{val}" for col in categorical_cols 
                                for val in df[col].unique()]
feature_importance = model.named_steps['regressor'].feature_importances_

# Try to match feature importance with feature names
# Note: This may not perfectly align due to OneHotEncoder's internal ordering
plt.figure(figsize=(10, 6))
try:
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.xticks(range(len(sorted_idx)), 
               [feature_names[i] if i < len(feature_names) else f"feature_{i}" 
                for i in sorted_idx], 
               rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
except Exception as e:
    print(f"Could not plot feature importance: {e}")

# Save the model
joblib.dump(model, 'customer_spend_model.pkl')
print("\nModel saved as 'customer_spend_model.pkl'")

# Example prediction function
def predict_spend(customer_data):
    """
    Predict spending for a single customer
    
    Args:
        customer_data (dict): Customer information with all required fields
        
    Returns:
        float: Predicted spending amount
    """
    # Convert to DataFrame with single row
    customer_df = pd.DataFrame([customer_data])
    
    # Make prediction
    prediction = model.predict(customer_df)[0]
    return prediction

# Example usage
print("\nExample prediction:")
new_customer = {
    'CustomerID': 9999,
    'Age': 38,
    'Income': 72000,
    'Gender': 'Male'
}
predicted_spend = predict_spend(new_customer)
print(f"Predicted spend for new customer: ${predicted_spend:.2f}")

