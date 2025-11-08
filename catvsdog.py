import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = r'C:\Users\Sindhu\Downloads\cats_vs_dogs\train'
IMG_SIZE = (128, 128)
MAX_IMAGES_PER_CLASS = 500
RANDOM_STATE = 42

# Global variables for model
trained_model = None
trained_scaler = None

# ==========================================
# FEATURE EXTRACTION USING HOG
# ==========================================
def extract_hog_features(image):
    """Extract Histogram of Oriented Gradients (HOG) features"""
    image_resized = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    win_size = IMG_SIZE
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    
    return features.flatten()

# ==========================================
# LOAD DATASET
# ==========================================
def load_dataset(dataset_path, max_per_class=500):
    """Load cat and dog images from a single folder"""
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path '{dataset_path}' does not exist!")
        print("Creating sample data for demonstration...")
        X = np.random.rand(100, 8100)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    files = os.listdir(dataset_path)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"Warning: No image files found in '{dataset_path}'!")
        print("Creating sample data for demonstration...")
        X = np.random.rand(100, 8100)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    print(f"Found {len(image_files)} images in dataset")
    print("Processing images...")
    
    cat_count = 0
    dog_count = 0
    
    for i, filename in enumerate(image_files):
        try:
            filename_lower = filename.lower()
            if 'cat' in filename_lower:
                label = 0
                if cat_count >= max_per_class:
                    continue
                cat_count += 1
            elif 'dog' in filename_lower:
                label = 1
                if dog_count >= max_per_class:
                    continue
                dog_count += 1
            else:
                continue
            
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            features = extract_hog_features(img)
            X.append(features)
            y.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")
            
        except Exception as e:
            continue
    
    print(f"\n✓ Loaded {cat_count} cat images")
    print(f"✓ Loaded {dog_count} dog images")
    
    return np.array(X), np.array(y)

# ==========================================
# TRAIN MODEL FUNCTION
# ==========================================
def train_model(dataset_path):
    """Train the SVM model"""
    global trained_model, trained_scaler
    
    print("=" * 60)
    print("         CAT VS DOG CLASSIFIER - SVM")
    print("=" * 60)
    
    print("\n[1/5] Loading dataset and extracting features...")
    X, y = load_dataset(dataset_path, MAX_IMAGES_PER_CLASS)
    
    if len(X) == 0:
        print("Error: No data loaded. Cannot train model.")
        return False
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Cats: {np.sum(y == 0)}")
    print(f"  Dogs: {np.sum(y == 1)}")
    print(f"  Feature vector size: {X.shape[1]}")
    print(f"{'='*60}")
    
    print("\n[2/5] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✓ Scaling complete!")
    
    trained_scaler = scaler
    
    print("\n[4/5] Training SVM classifier...")
    print("  This may take a few minutes...")
    svm_classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
    svm_classifier.fit(X_train_scaled, y_train)
    print("  ✓ Training complete!")
    
    trained_model = svm_classifier
    
    print("\n[5/5] Evaluating model...")
    y_pred_test = svm_classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("         MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Cat', 'Dog']))
    
    return True

# ==========================================
# PREDICTION FUNCTION WITH IMAGE DISPLAY
# ==========================================
def predict_and_show_image(image_path):
    """
    Predict and display the image with result
    
    Parameters:
    - image_path: Path to the image file
    
    Returns:
    - result: "Cat" or "Dog"
    """
    global trained_model, trained_scaler
    
    if trained_model is None or trained_scaler is None:
        print("Error: Model not trained yet. Please train the model first.")
        return None
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Convert BGR to RGB for proper display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features = extract_hog_features(img)
        features_scaled = trained_scaler.transform([features])
        
        # Predict
        prediction = trained_model.predict(features_scaled)[0]
        result = "Dog" if prediction == 1 else "Cat"
        emoji = "🐶" if prediction == 1 else "🐱"
        
        # Display image with prediction
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'Prediction: {result} {emoji}', 
                  fontsize=24, fontweight='bold', 
                  color='green' if result == 'Cat' else 'blue',
                  pad=20)
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*60}")
        print(f"  PREDICTION RESULT: {result} {emoji}")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

# ==========================================
# BATCH PREDICTION FOR MULTIPLE IMAGES
# ==========================================
def predict_multiple_images(image_paths):
    """
    Predict multiple images and display them in a grid
    
    Parameters:
    - image_paths: List of image file paths
    """
    global trained_model, trained_scaler
    
    if trained_model is None or trained_scaler is None:
        print("Error: Model not trained yet. Please train the model first.")
        return
    
    num_images = len(image_paths)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, image_path in enumerate(image_paths):
        try:
            # Load and predict
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features = extract_hog_features(img)
            features_scaled = trained_scaler.transform([features])
            prediction = trained_model.predict(features_scaled)[0]
            
            result = "Dog" if prediction == 1 else "Cat"
            emoji = "🐶" if prediction == 1 else "🐱"
            
            # Display
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f'{result} {emoji}', 
                               fontsize=16, fontweight='bold',
                               color='green' if result == 'Cat' else 'blue')
            
            print(f"Image {idx+1}: {result} {emoji}")
            
        except Exception as e:
            print(f"Error processing image {idx+1}: {str(e)}")
            continue
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# INTERACTIVE IMAGE SELECTOR
# ==========================================
def interactive_image_classifier():
    """
    Interactive function to select and classify images
    """
    global trained_model, trained_scaler
    
    if trained_model is None or trained_scaler is None:
        print("Error: Model not trained yet. Please train the model first.")
        return
    
    print("\n" + "=" * 60)
    print("         INTERACTIVE IMAGE CLASSIFIER")
    print("=" * 60)
    print("\nOptions:")
    print("1. Classify a single image")
    print("2. Classify multiple images")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter the full path to the image: ").strip()
            if os.path.exists(image_path):
                predict_and_show_image(image_path)
            else:
                print(f"Error: File not found at {image_path}")
        
        elif choice == '2':
            num_images = int(input("How many images do you want to classify? "))
            image_paths = []
            for i in range(num_images):
                path = input(f"Enter path for image {i+1}: ").strip()
                if os.path.exists(path):
                    image_paths.append(path)
                else:
                    print(f"Warning: File not found at {path}, skipping...")
            
            if image_paths:
                predict_multiple_images(image_paths)
            else:
                print("No valid image paths provided.")
        
        elif choice == '3':
            print("Exiting. Thank you!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Cat vs Dog Classifier\n")
    
    # Train the model
    success = train_model(DATASET_PATH)
    
    if success:
        print("\n" + "=" * 60)
        print("         🎉 MODEL TRAINING COMPLETE!")
        print("=" * 60)
        
        # Example: Predict a single image
        print("\n📸 EXAMPLE: Single Image Prediction")
        print("-" * 60)
        
        # Method 1: Direct function call
        # Uncomment and update the path to your image
        # predict_and_show_image(r'C:\path\to\your\cat_or_dog_image.jpg')
        
        # Method 2: Interactive mode
        print("\nStarting interactive classifier...")
        interactive_image_classifier()
        
    else:
        print("\n❌ Model training failed. Please check your dataset path.")
        print(f"Dataset path: {DATASET_PATH}")
        print("\nMake sure your dataset folder contains images with 'cat' or 'dog' in the filename.")
