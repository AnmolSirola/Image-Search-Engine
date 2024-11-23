import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
from pathlib import Path
import os
import logging
import shutil
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class to store search results"""
    distance: float
    image_path: str
    metadata: dict = None

class FeatureExtractor:
    """Class to handle feature extraction from images using VGG16"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the feature extractor with VGG16 model
        
        Args:
            model_path: Optional path to saved model weights
        """
        self.model = VGG16(weights='imagenet', include_top=True)
        # Remove the last layer (classification layer)
        self.model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('fc2').output)
        self.input_shape = (224, 224)  # VGG16 input shape
        self.batch_size = 32
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            logger.info(f"Loaded model weights from {model_path}")
    
    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess image for VGG16
        
        Args:
            img: PIL Image object
        
        Returns:
            Preprocessed image as numpy array
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        img.thumbnail(self.input_shape, Image.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', self.input_shape, (255, 255, 255))
        new_img.paste(img, ((self.input_shape[0] - img.size[0]) // 2,
                           (self.input_shape[1] - img.size[1]) // 2))
        
        # Convert to array and preprocess
        img_array = img_to_array(new_img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def extract(self, img: Image.Image) -> np.ndarray:
        """
        Extract features from image
        
        Args:
            img: PIL Image object
        
        Returns:
            Feature vector as numpy array
        """
        try:
            preprocessed_img = self.preprocess_image(img)
            features = self.model.predict(preprocessed_img)
            normalized_features = features / np.linalg.norm(features)
            return normalized_features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

class ImageSearchEngine:
    """Class to handle image search operations"""
    
    def __init__(self, feature_dir: str, img_dir: str, cache_dir: str = "cache"):
        """
        Initialize the search engine
        
        Args:
            feature_dir: Directory containing feature vectors
            img_dir: Directory containing images
            cache_dir: Directory for caching results
        """
        self.feature_dir = Path(feature_dir)
        self.img_dir = Path(img_dir)
        self.cache_dir = Path(cache_dir)
        self.feature_extractor = FeatureExtractor()
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load features and image paths
        self.features = []
        self.img_paths = []
        self.load_features()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized search engine with {len(self.features)} images")
    
    def load_features(self):
        """Load all feature vectors and image paths"""
        try:
            for feature_path in self.feature_dir.glob("*.npy"):
                self.features.append(np.load(feature_path))
                img_path = self.img_dir / (feature_path.stem + ".jpg")
                if img_path.exists():
                    self.img_paths.append(img_path)
                else:
                    logger.warning(f"Image not found for feature: {feature_path}")
            
            self.features = np.array(self.features)
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise

    def search(self, query_img: Image.Image, top_k: int = 30) -> List[SearchResult]:
        """
        Search for similar images
        
        Args:
            query_img: Query image as PIL Image
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Extract features from query image
            query_features = self.feature_extractor.extract(query_img)
            
            # Calculate distances
            distances = np.linalg.norm(self.features - query_features, axis=1)
            
            # Get top k results
            top_indices = np.argsort(distances)[:top_k]
            
            # Create search results
            results = []
            for idx in top_indices:
                result = SearchResult(
                    distance=float(distances[idx]),
                    image_path=str(self.img_paths[idx]),
                    metadata=self._get_image_metadata(self.img_paths[idx])
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def _get_image_metadata(self, img_path: Path) -> dict:
        """Get metadata for an image"""
        try:
            img = Image.open(img_path)
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "filename": img_path.name
            }
        except Exception as e:
            logger.warning(f"Error getting metadata for {img_path}: {str(e)}")
            return {}

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize search engine
search_engine = ImageSearchEngine(
    feature_dir="./static/feature",
    img_dir="./static/img",
    cache_dir="./static/cache"
)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle main page requests"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'query_img' not in request.files:
                return render_template('index.html', error="No file uploaded")
            
            file = request.files['query_img']
            if file.filename == '':
                return render_template('index.html', error="No file selected")
            
            # Process uploaded image
            img = Image.open(file.stream)
            
            # Save query image
            timestamp = datetime.now().isoformat().replace(":", ".")
            uploaded_img_path = f"static/uploaded/{timestamp}_{file.filename}"
            img.save(uploaded_img_path)
            
            # Perform search
            results = search_engine.search(img, top_k=30)
            
            return render_template('index.html',
                                 query_path=uploaded_img_path,
                                 results=results)
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for image search"""
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        results = search_engine.search(img, top_k=30)
        
        # Convert results to JSON-serializable format
        response = {
            'results': [
                {
                    'distance': result.distance,
                    'image_path': result.image_path,
                    'metadata': result.metadata
                }
                for result in results
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return render_template('index.html', error="File too large (max 16MB)"), 413

if __name__ == "__main__":
    # Create required directories
    os.makedirs("static/uploaded", exist_ok=True)
    os.makedirs("static/feature", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)
    os.makedirs("static/cache", exist_ok=True)
    
    # Start the application
    app.run(host="0.0.0.0", port=5000, debug=True)