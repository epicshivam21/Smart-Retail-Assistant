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
   #Prepare train and validation data generators with augmentation
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
    #Train the image classification model
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
    #Test inference with the trained model
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
        'Preprocess image for the model'
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
        #Classify an image
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
        "Get list of classes the model can recognize"
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


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Define model path
MODEL_PATH = 'customer_spend_model.pkl'

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Function to load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Load the dataset
file_path = "D:/aicte/Dataset/archive_customer/Mall_Customers.csv"
df = load_dataset(file_path)

if df is None:
    print("Exiting due to dataset loading error")
    exit()

# Data exploration and visualization
print(f"Dataset shape: {df.shape}")
print("\nData overview:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# Create new features through feature engineering
print("\nPerforming feature engineering...")

# Make sure Age and Income are numeric before feature engineering
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')

# 1. Age groups
df['Age_Group'] = pd.cut(
    df['Age'], 
    bins=[0, 25, 35, 50, 100], 
    labels=['Young Adult', 'Adult', 'Middle Age', 'Senior']
)

# 2. Income tiers - handle potential errors with q parameter
try:
    df['Income_Tier'] = pd.qcut(
        df['Income'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
except ValueError as e:
    print(f"Warning: Could not create Income_Tier: {e}")
    # Alternative: Use regular cut if qcut fails
    income_ranges = [df['Income'].min()-1, df['Income'].quantile(0.25), 
                    df['Income'].quantile(0.5), df['Income'].quantile(0.75), 
                    df['Income'].max()+1]
    df['Income_Tier'] = pd.cut(
        df['Income'],
        bins=income_ranges,
        labels=['Low', 'Medium', 'High', 'Very High']
    )

# 3. Age-Income interaction feature
df['Age_Income_Factor'] = df['Age'] * df['Income'] / 1000

# Visualize key relationships
plt.figure(figsize=(12, 10))

# Plot 1: Distribution of the target variable
plt.subplot(2, 2, 1)
sns.histplot(df['Spending'], kde=True, color='darkblue')
plt.title('Distribution of Customer Spending', fontsize=12, fontweight='bold')
plt.xlabel('Spending Amount', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Plot 2: Spending by Gender
plt.subplot(2, 2, 2)
sns.boxplot(x='Gender', y='Spending', hue='Gender', data=df, palette=['lightblue', 'pink'], legend=False)
plt.title('Spending by Gender', fontsize=12, fontweight='bold')
plt.xlabel('Gender', fontsize=10)
plt.ylabel('Spending Amount', fontsize=10)

# Plot 3: Spending vs Age
plt.subplot(2, 2, 3)
sns.scatterplot(x='Age', y='Spending', hue='Gender', data=df, alpha=0.7)
plt.title('Spending vs Age', fontsize=12, fontweight='bold')
plt.xlabel('Age', fontsize=10)
plt.ylabel('Spending Amount', fontsize=10)

# Plot 4: Spending vs Income
plt.subplot(2, 2, 4)
sns.scatterplot(x='Income', y='Spending', hue='Gender', data=df, alpha=0.7)
plt.title('Spending vs Income', fontsize=12, fontweight='bold')
plt.xlabel('Income', fontsize=10)
plt.ylabel('Spending Amount', fontsize=10)

plt.tight_layout()
plt.savefig('customer_spending_analysis.png', dpi=300)
plt.show()
plt.close()

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Separate features and target
X = df.drop(['Spending', 'Age_Group', 'Income_Tier'], axis=1)  # Drop derived categorical features
y = df['Spending']

# Identify categorical and numerical columns
categorical_cols = ['Gender']
numerical_cols = ['Age', 'Income', 'Age_Income_Factor']

# Check for outliers in numerical features and handle them
def handle_outliers(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Print outlier info
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"Column {col}: {len(outliers)} outliers detected")
        
        # Cap outliers instead of removing
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

print("\nChecking for outliers...")
df_clean = handle_outliers(df, numerical_cols[:2])  # Only handle Age and Income outliers

# Update X with cleaned data
cols_to_drop = ['Spending']
if 'Age_Group' in df_clean.columns:
    cols_to_drop.append('Age_Group')
if 'Income_Tier' in df_clean.columns:
    cols_to_drop.append('Income_Tier')

X = df_clean.drop(cols_to_drop, axis=1)

# Data preprocessing
print("\nSetting up preprocessing pipeline...")
available_num_cols = [col for col in numerical_cols if col in X.columns]
available_cat_cols = [col for col in categorical_cols if col in X.columns]

print(f"Using numerical columns: {available_num_cols}")
print(f"Using categorical columns: {available_cat_cols}")

transformers = []
if available_num_cols:
    transformers.append(('num', StandardScaler(), available_num_cols))
if available_cat_cols:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat_cols))

preprocessor = ColumnTransformer(
    transformers=transformers, 
    remainder='passthrough')

# Split data into training and testing sets
if 'Age_Group' in df_clean.columns and len(df_clean['Age_Group'].unique()) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df_clean['Age_Group']
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define models to try
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Training predictions
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Training - RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
    print(f"Testing  - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}")
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Actual vs Predicted (Training)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    axes[0, 0].set_title(f'Training: Actual vs Predicted ({model_name})', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Actual Spending', fontsize=10)
    axes[0, 0].set_ylabel('Predicted Spending', fontsize=10)
    axes[0, 0].text(0.05, 0.95, f'R² = {train_r2:.2f}\nRMSE = {train_rmse:.2f}', 
                 transform=axes[0, 0].transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Actual vs Predicted (Testing)
    axes[0, 1].scatter(y_test, y_pred, alpha=0.5, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 1].set_title(f'Testing: Actual vs Predicted ({model_name})', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Spending', fontsize=10)
    axes[0, 1].set_ylabel('Predicted Spending', fontsize=10)
    axes[0, 1].text(0.05, 0.95, f'R² = {test_r2:.2f}\nRMSE = {test_rmse:.2f}', 
                 transform=axes[0, 1].transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Residuals (Testing)
    residuals = y_test - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='purple')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title(f'Residual Plot ({model_name})', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Spending', fontsize=10)
    axes[1, 0].set_ylabel('Residuals', fontsize=10)
    
    # Plot 4: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1, 1], color='darkgreen')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_title(f'Residual Distribution ({model_name})', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Residual Value', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_performance_evaluation.png', dpi=300)
    plt.show()
    plt.close()
    
    return test_mse, test_rmse, test_mae, test_r2, y_pred

# Train and evaluate models
best_r2 = -float('inf')
best_model = None
best_model_name = None
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Create pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipe.fit(X_train, y_train)
    
    # Evaluate
    mse, rmse, mae, r2, y_pred = evaluate_model(pipe, X_train, X_test, y_train, y_test, model_name)
    results[model_name] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    # Update best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipe
        best_model_name = model_name

# Create comparative performance plot
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[model]['rmse'] for model in results],
    'MAE': [results[model]['mae'] for model in results],
    'R²': [results[model]['r2'] for model in results]
})

# Melt the dataframe for easier plotting
melted_df = pd.melt(performance_df, id_vars=['Model'], var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Metric', title_fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison.png', dpi=300)
plt.show()
plt.close()

print(f"\nBest performing model: {best_model_name} with R² = {best_r2:.2f}")

# Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'RandomForest':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }
else:  # GradientBoosting
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7]
    }

# Create grid search
grid_search = GridSearchCV(
    best_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters
print(f"\nBest parameters: {grid_search.best_params_}")

# Create plot of grid search results if model is RandomForest
if best_model_name == 'RandomForest':
    # Extract results from CV
    cv_results = grid_search.cv_results_
    
    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'n_estimators': cv_results['param_regressor__n_estimators'],
        'max_depth': cv_results['param_regressor__max_depth'],
        'min_samples_split': cv_results['param_regressor__min_samples_split'],
        'mean_test_score': cv_results['mean_test_score'],
        'std_test_score': cv_results['std_test_score']
    })
    
    # Replace None with 'None' for plotting
    results_df['max_depth'] = results_df['max_depth'].fillna('None').astype(str)
    
    # Create grouped bar plots for n_estimators and max_depth
    plt.figure(figsize=(14, 8))
    
    # Group by n_estimators and max_depth
    grouped_results = results_df.groupby(['n_estimators', 'max_depth'])['mean_test_score'].mean().reset_index()
    
    # Pivot for plotting
    pivot_df = grouped_results.pivot(index='n_estimators', columns='max_depth', values='mean_test_score')
    
    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('RandomForest: Effect of n_estimators and max_depth on R²', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('Mean R² Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Max Depth', title_fontsize=10)
    
    # Annotate best combination
    best_n_est = grid_search.best_params_['regressor__n_estimators']
    best_max_depth = str(grid_search.best_params_['regressor__max_depth'])
    best_samples_split = grid_search.best_params_['regressor__min_samples_split']
    best_score = grid_search.best_score_
    
    plt.annotate(f'Best: n_est={best_n_est}, max_depth={best_max_depth},\n'
                f'min_samples_split={best_samples_split}, R²={best_score:.2f}',
                xy=(0.5, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('randomforest_hyperparameter_tuning.png', dpi=300)
    plt.show()
    plt.close()

# Evaluate tuned model
tuned_model = grid_search.best_estimator_
mse, rmse, mae, r2, y_pred = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")

# Compare actual vs predicted values with confidence intervals
plt.figure(figsize=(10, 8))

# Sort test data by actual values for better visualization
sorted_indices = np.argsort(y_test.values)
sorted_actual = y_test.values[sorted_indices]
sorted_pred = y_pred[sorted_indices]

# Plot actual vs predicted with index
plt.plot(range(len(sorted_actual)), sorted_actual, 'b-', label='Actual')
plt.plot(range(len(sorted_pred)), sorted_pred, 'r--', label='Predicted')

# Add confidence interval
if best_model_name == 'RandomForest':
    # For RandomForest, we can use the standard deviation of predictions across trees
    preds = []
    for estimator in tuned_model.named_steps['regressor'].estimators_:
        # Create a new pipeline with the preprocessor and a single tree
        temp_pipeline = Pipeline([
            ('preprocessor', tuned_model.named_steps['preprocessor']),
            ('tree', estimator)
        ])
        preds.append(temp_pipeline.predict(X_test))
    
    preds = np.array(preds)
    std_devs = np.std(preds, axis=0)
    
    # Sort the standard deviations according to actual values
    sorted_stds = std_devs[sorted_indices]
    
    # Plot confidence intervals (±1 std dev)
    plt.fill_between(range(len(sorted_pred)), 
                     sorted_pred - sorted_stds,
                     sorted_pred + sorted_stds,
                     color='r', alpha=0.2, label='±1 Std Dev')

plt.title('Actual vs Predicted Customer Spending with Uncertainty', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index (sorted by actual value)', fontsize=12)
plt.ylabel('Spending', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_with_uncertainty.png', dpi=300)
plt.show()
plt.close()

# Feature importance analysis
if best_model_name == 'RandomForest':
    feature_importances = tuned_model.named_steps['regressor'].feature_importances_
else:
    feature_importances = tuned_model.named_steps['regressor'].feature_importances_

# Get feature names from preprocessor
try:
    preprocessor = tuned_model.named_steps['preprocessor']
    feature_names = []
    
    # Get numerical feature names if they exist
    num_transformer_idx = None
    cat_transformer_idx = None
    
    # Find indices for transformers
    for i, (name, _, _) in enumerate(preprocessor.transformers_):
        if name == 'num':
            num_transformer_idx = i
        elif name == 'cat':
            cat_transformer_idx = i
    
    # Add numerical feature names
    if num_transformer_idx is not None:
        feature_names.extend(preprocessor.transformers_[num_transformer_idx][2])
    
    # Get one-hot encoded feature names
    if cat_transformer_idx is not None:
        ohe_categories = preprocessor.transformers_[cat_transformer_idx][1].categories_
        cat_cols = preprocessor.transformers_[cat_transformer_idx][2]
        for i, category in enumerate(cat_cols):
            for cat_value in ohe_categories[i]:
                feature_names.append(f"{category}_{cat_value}")

    print(f"Extracted {len(feature_names)} feature names")
except Exception as e:
    print(f"Error extracting feature names: {e}")
    # Fallback to generic feature names
    feature_names = [f"feature_{i}" for i in range(len(feature_importances))]

# Create improved feature importance visualization
plt.figure(figsize=(12, 8))
try:
    # Make sure we don't have index issues
    min_length = min(len(feature_names), len(feature_importances))
    print(f"Using {min_length} features for importance plot")
    
    # Create a dataframe for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names[:min_length],
        'Importance': feature_importances[:min_length]
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot - using hue parameter to avoid deprecation warning
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='viridis', legend=False)
    plt.title('Feature Importance for Customer Spending Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage annotations
    total = importance_df['Importance'].sum()
    for i, (index, row) in enumerate(importance_df.iterrows()):
        plt.text(row['Importance'] + 0.01, i, f"{row['Importance']/total:.1%}", 
                 va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance_improved.png', dpi=300)
    print("Feature importance plot saved successfully")
except Exception as e:
    print(f"Error creating feature importance plot: {e}")
finally:
    plt.show()
    plt.close()

# Save the tuned model
joblib.dump(tuned_model, 'customer_spend_model_tuned.pkl')
print("\nTuned model saved as 'customer_spend_model_tuned.pkl'")

# Function to visualize feature relationships with spending
def plot_feature_relationships(df, best_model_name):
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Age vs Spending with trend line
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='Age', y='Spending', data=df, hue='Gender', alpha=0.6)
    # Add trend line
    sns.regplot(x='Age', y='Spending', data=df, scatter=False, color='red')
    plt.title('Age vs Spending with Trend', fontsize=12, fontweight='bold')
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Spending', fontsize=10)
    
    # Plot 2: Income vs Spending with trend line
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='Income', y='Spending', data=df, hue='Gender', alpha=0.6)
    # Add trend line
    sns.regplot(x='Income', y='Spending', data=df, scatter=False, color='red')
    plt.title('Income vs Spending with Trend', fontsize=12, fontweight='bold')
    plt.xlabel('Income', fontsize=10)
    plt.ylabel('Spending', fontsize=10)
    
    # Plot 3: Spending by Age Group
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Age_Group', y='Spending', hue='Age_Group', data=df, palette='viridis', legend=False)
    plt.title('Spending by Age Group', fontsize=12, fontweight='bold')
    plt.xlabel('Age Group', fontsize=10)
    plt.ylabel('Spending', fontsize=10)
    plt.xticks(rotation=45)
    
    # Plot 4: Spending by Income Tier
    plt.subplot(2, 2, 4)
    sns.boxplot(x='Income_Tier', y='Spending', hue='Income_Tier', data=df, palette='viridis', legend=False)
    plt.title('Spending by Income Tier', fontsize=12, fontweight='bold')
    plt.xlabel('Income Tier', fontsize=10)
    plt.ylabel('Spending', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{best_model_name}_feature_relationships.png', dpi=300)
    plt.show()
    plt.close()


# Create feature relationship plots
plot_feature_relationships(df, best_model_name)

# Example prediction function
def predict_spend(customer_data, model):
    #Predict spending for a single customer
   
    # Ensure all required columns are present
    required_cols = ['CustomerID', 'Gender', 'Age', 'Income']
    for col in required_cols:
        if col not in customer_data:
            raise ValueError(f"Missing required column: {col}")
    
    # Make a copy to avoid modifying the original
    customer_data_copy = customer_data.copy()
    
    # Convert types to ensure compatibility
    if 'Age' in customer_data_copy:
        customer_data_copy['Age'] = float(customer_data_copy['Age'])
    if 'Income' in customer_data_copy:
        customer_data_copy['Income'] = float(customer_data_copy['Income'])
    
    # Add derived features
    if 'Age_Income_Factor' not in customer_data_copy and 'Age' in customer_data_copy and 'Income' in customer_data_copy:
        customer_data_copy['Age_Income_Factor'] = customer_data_copy['Age'] * customer_data_copy['Income'] / 1000
    
    # Convert to DataFrame with single row
    customer_df = pd.DataFrame([customer_data_copy])
    
    # Print column types for debugging
    print("Prediction dataframe columns:", customer_df.columns.tolist())
    print("Prediction dataframe types:", customer_df.dtypes)
    
    # Make prediction
    try:
        prediction = model.predict(customer_df)[0]
        return max(0, prediction)  # Ensure prediction is not negative
    except Exception as e:
        print(f"Prediction error: {e}")
        # Print the pipeline expected features for debugging
        try:
            print("Model expected features:", model.feature_names_in_)
        except AttributeError:
            pass
        return None

# Function to analyze prediction factors
def explain_prediction(customer_data, model, feature_names, feature_importances):
    #Explain what factors influenced the prediction most

    # Check if feature_names and feature_importances have valid data
    if not feature_names or len(feature_names) == 0:
        print("Warning: No feature names available for explanation")
        return [], []
        
    if feature_importances is None or len(feature_importances) == 0:
        print("Warning: No feature importance data available for explanation")
        return [], []
    
    # Make sure lengths match and are valid
    min_length = min(len(feature_names), len(feature_importances))
    if min_length == 0:
        print("Warning: Either feature names or importances have zero length")
        return [], []
        
    print(f"Explaining prediction using {min_length} features")
    
    # Create DataFrame with customer data
    df = pd.DataFrame([customer_data])

