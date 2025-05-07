#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
    


# In[ ]:




