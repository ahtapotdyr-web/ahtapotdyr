"""
Advanced Facial Recognition & Web Search Engine
Complete production-ready system
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import hashlib
from pathlib import Path

# Web Framework
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# Face Recognition
import face_recognition
import cv2
import numpy as np
from PIL import Image as PILImage

# Search Engines
import aiohttp
import requests
from bs4 import BeautifulSoup

# Data Processing
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///facial_recognition.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # Face Recognition settings
    FACE_DISTANCE_THRESHOLD = 0.6
    MIN_FACE_SIZE = 20
    USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'
    MODEL = 'hog'  # or 'cnn' for GPU
    
    # Search settings
    MAX_SEARCH_RESULTS = 100
    SEARCH_TIMEOUT = 30
    
    # APIs
    BING_API_KEY = os.getenv('BING_API_KEY', '')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    GOOGLE_ENGINE_ID = os.getenv('GOOGLE_ENGINE_ID', '')

# ============================================================================
# DATABASE MODELS
# ============================================================================

db = SQLAlchemy()

class ImageRecord(db.Model):
    """Store uploaded images and their face encodings"""
    __tablename__ = 'images'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False, unique=True)
    
    # Image metadata
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(50))
    
    # Face data (stored as JSON)
    face_count = db.Column(db.Integer, default=0)
    face_encodings = db.Column(db.Text)  # JSON serialized
    face_locations = db.Column(db.Text)  # JSON serialized
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by = db.Column(db.String(100), default='anonymous')
    
    # Relations
    search_results = db.relationship('SearchResultRecord', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.original_name,
            'width': self.width,
            'height': self.height,
            'face_count': self.face_count,
            'created_at': self.created_at.isoformat(),
            'file_size': self.file_size
        }

class SearchHistoryRecord(db.Model):
    """Store search history"""
    __tablename__ = 'search_history'
    
    id = db.Column(db.Integer, primary_key=True)
    search_type = db.Column(db.String(50))  # 'face', 'reverse_image', 'combined'
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'))
    query_params = db.Column(db.Text)  # JSON
    results_count = db.Column(db.Integer, default=0)
    search_duration = db.Column(db.Float)
    status = db.Column(db.String(20), default='completed')
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    search_results = db.relationship('SearchResultRecord', cascade='all, delete-orphan')

class SearchResultRecord(db.Model):
    """Store individual search results"""
    __tablename__ = 'search_results'
    
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.Integer, db.ForeignKey('search_history.id'))
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'))
    
    # Result data
    url = db.Column(db.Text, nullable=False)
    title = db.Column(db.String(500))
    description = db.Column(db.Text)
    thumbnail = db.Column(db.String(500))
    source = db.Column(db.String(100))  # 'bing', 'google', 'face_db'
    
    # Match info
    match_score = db.Column(db.Float)
    match_type = db.Column(db.String(50))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============================================================================
# FACE RECOGNITION ENGINE
# ============================================================================

class FaceRecognitionEngine:
    """Advanced face recognition with caching and batch processing"""
    
    def __init__(self, use_gpu=False, model='hog', cache_file='face_cache.pkl'):
        self.use_gpu = use_gpu
        self.model = model if (model == 'hog' or use_gpu) else 'hog'
        self.cache = {}
        self.cache_file = cache_file
        self.load_cache()
        logger.info(f"FaceRecognitionEngine initialized: model={self.model}, gpu={use_gpu}")
    
    def load_cache(self):
        """Load face encoding cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.cache)} cached face encodings")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def save_cache(self):
        """Save face encoding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
                logger.info(f"Saved {len(self.cache)} face encodings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def detect_and_encode_faces(self, image_path: str) -> Tuple[List, List]:
        """
        Detect faces and return their locations and encodings
        """
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model=self.model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            logger.info(f"Detected {len(face_locations)} faces in {image_path}")
            return face_locations, face_encodings
        
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return [], []
    
    def compare_faces_fast(self, known_encoding: np.ndarray,
                          test_encodings: List[np.ndarray],
                          tolerance: float = 0.6) -> List[Dict]:
        """
        Fast batch comparison using numpy
        """
        results = []
        distances = face_recognition.face_distance([known_encoding], test_encodings)
        
        for idx, distance in enumerate(distances):
            similarity = 1 - distance
            is_match = distance <= tolerance
            
            results.append({
                'index': idx,
                'distance': float(distance),
                'similarity': float(similarity),
                'is_match': bool(is_match)
            })
        
        return results
    
    def extract_face(self, image_path: str, face_idx: int = 0,
                    padding: int = 20) -> PILImage.Image:
        """Extract face region from image"""
        try:
            pil_image = PILImage.open(image_path)
            face_locations, _ = self.detect_and_encode_faces(image_path)
            
            if not face_locations or face_idx >= len(face_locations):
                return None
            
            top, right, bottom, left = face_locations[face_idx]
            
            # Add padding
            top = max(0, top - padding)
            left = max(0, left - padding)
            right = min(pil_image.width, right + padding)
            bottom = min(pil_image.height, bottom + padding)
            
            return pil_image.crop((left, top, right, bottom))
        
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def get_face_statistics(self, image_path: str) -> Dict:
        """Get detailed face statistics"""
        try:
            pil_image = PILImage.open(image_path)
            face_locations, face_encodings = self.detect_and_encode_faces(image_path)
            
            stats = {
                'total_faces': len(face_locations),
                'image_size': (pil_image.width, pil_image.height),
                'image_area': pil_image.width * pil_image.height,
                'faces': []
            }
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_area = (right - left) * (bottom - top)
                coverage = (face_area / stats['image_area']) * 100
                
                stats['faces'].append({
                    'index': i,
                    'box': [top, right, bottom, left],
                    'width': right - left,
                    'height': bottom - top,
                    'area': face_area,
                    'coverage_pct': coverage
                })
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting face statistics: {e}")
            return {}

# ============================================================================
# SEARCH ENGINE
# ============================================================================

class AdvancedSearchEngine:
    """Multi-source search engine for reverse image and web search"""
    
    def __init__(self, bing_key='', google_key='', google_engine_id=''):
        self.bing_key = bing_key
        self.google_key = google_key
        self.google_engine_id = google_engine_id
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        logger.info("AdvancedSearchEngine initialized")
    
    async def search_bing(self, query: str, count: int = 50) -> List[Dict]:
        """Search Bing Images API"""
        try:
            url = "https://api.bing.microsoft.com/v7.0/images/search"
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_key,
                'User-Agent': self.headers['User-Agent']
            }
            
            params = {
                'q': query,
                'count': min(count, 150),
                'mkt': 'en-US',
                'safeSearch': 'Moderate'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        
                        for item in data.get('value', [])[:count]:
                            results.append({
                                'source': 'bing',
                                'url': item.get('contentUrl'),
                                'title': item.get('name'),
                                'thumbnail': item.get('thumbnailUrl'),
                                'description': item.get('hostPageDisplayUrl'),
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        
                        logger.info(f"Found {len(results)} Bing results for '{query}'")
                        return results
                    
                    logger.error(f"Bing API error: {resp.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error searching Bing: {e}")
            return []
    
    async def search_google(self, query: str, count: int = 50) -> List[Dict]:
        """Search Google Custom Search API"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            
            params = {
                'q': query,
                'cx': self.google_engine_id,
                'key': self.google_key,
                'searchType': 'image',
                'num': min(count, 10),
                'safe': 'medium'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        
                        for item in data.get('items', [])[:count]:
                            results.append({
                                'source': 'google',
                                'url': item.get('link'),
                                'title': item.get('title'),
                                'thumbnail': item.get('image', {}).get('thumbnailUrl'),
                                'description': item.get('snippet'),
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        
                        logger.info(f"Found {len(results)} Google results for '{query}'")
                        return results
                    
                    logger.error(f"Google API error: {resp.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error searching Google: {e}")
            return []
    
    async def search_multiple_sources(self, query: str, count: int = 100) -> List[Dict]:
        """Search multiple engines in parallel"""
        tasks = []
        
        if self.bing_key:
            tasks.append(self.search_bing(query, count // 2))
        
        if self.google_key:
            tasks.append(self.search_google(query, count // 2))
        
        if not tasks:
            logger.warning("No search APIs configured")
            return []
        
        try:
            results = await asyncio.gather(*tasks)
            combined = []
            
            for result_list in results:
                combined.extend(result_list)
            
            # Remove duplicates
            seen_urls = set()
            unique = []
            
            for result in combined:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique.append(result)
            
            return unique[:count]
        
        except Exception as e:
            logger.error(f"Error in parallel search: {e}")
            return []
    
    def scrape_webpage(self, url: str) -> Dict:
        """Scrape webpage for metadata"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            data = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'description': '',
                'images': [],
                'text': ''
            }
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                data['description'] = meta_desc.get('content', '')
            
            # Get images
            for img in soup.find_all('img', limit=10):
                src = img.get('src', '')
                if src:
                    data['images'].append({
                        'url': src,
                        'alt': img.get('alt', '')
                    })
            
            # Get text
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs[:5]])
            data['text'] = text[:500]
            
            logger.info(f"Scraped {url}: {len(data['images'])} images found")
            return data
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {'url': url, 'error': str(e)}

# ============================================================================
# FLASK APPLICATION
# ============================================================================

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize engines
    face_engine = FaceRecognitionEngine(
        use_gpu=app.config['USE_GPU'],
        model=app.config['MODEL']
    )
    
    search_engine = AdvancedSearchEngine(
        bing_key=app.config['BING_API_KEY'],
        google_key=app.config['GOOGLE_API_KEY'],
        google_engine_id=app.config['GOOGLE_ENGINE_ID']
    )
    
    with app.app_context():
        db.create_all()
    
    # ========================================================================
    # API ROUTES
    # ========================================================================
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check"""
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'face_recognition': 'active',
                'search_engine': 'active',
                'database': 'active'
            }
        })
    
    @app.route('/api/upload', methods=['POST'])
    def upload_image():
        """Upload and process image"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Validate extension
            if not any(file.filename.lower().endswith(f'.{ext}') 
                      for ext in app.config['ALLOWED_EXTENSIONS']):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process with face recognition
            face_locations, face_encodings = face_engine.detect_and_encode_faces(filepath)
            stats = face_engine.get_face_statistics(filepath)
            
            # Get image info
            pil_img = PILImage.open(filepath)
            
            # Save to database
            record = ImageRecord(
                filename=filename,
                original_name=file.filename,
                filepath=filepath,
                width=pil_img.width,
                height=pil_img.height,
                file_size=os.path.getsize(filepath),
                mime_type=file.content_type,
                face_count=len(face_locations),
                face_encodings=json.dumps([e.tolist() for e in face_encodings]),
                face_locations=json.dumps([list(loc) for loc in face_locations])
            )
            
            db.session.add(record)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'image_id': record.id,
                'filename': filename,
                'face_count': len(face_locations),
                'statistics': stats
            })
        
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search/faces', methods=['POST'])
    def search_faces():
        """Search similar faces in database"""
        try:
            data = request.get_json()
            image_id = data.get('image_id')
            face_idx = data.get('face_index', 0)
            tolerance = data.get('tolerance', Config.FACE_DISTANCE_THRESHOLD)
            
            # Get reference image
            ref_image = ImageRecord.query.get(image_id)
            if not ref_image or not ref_image.face_encodings:
                return jsonify({'error': 'Image or encodings not found'}), 404
            
            ref_encodings = json.loads(ref_image.face_encodings)
            if face_idx >= len(ref_encodings):
                return jsonify({'error': 'Face index out of range'}), 400
            
            ref_encoding = np.array(ref_encodings[face_idx])
            
            # Search database
            start = time.time()
            matches = []
            
            other_images = ImageRecord.query.filter(ImageRecord.id != image_id).all()
            
            for other in other_images:
                if not other.face_encodings:
                    continue
                
                other_encodings = json.loads(other.face_encodings)
                results = face_engine.compare_faces_fast(
                    ref_encoding,
                    [np.array(e) for e in other_encodings],
                    tolerance
                )
                
                for result in results:
                    if result['is_match']:
                        matches.append({
                            'image_id': other.id,
                            'filename': other.original_name,
                            'similarity': result['similarity'],
                            'distance': result['distance']
                        })
            
            duration = time.time() - start
            
            # Save search history
            history = SearchHistoryRecord(
                search_type='face',
                image_id=image_id,
                query_params=json.dumps({'tolerance': tolerance, 'face_index': face_idx}),
                results_count=len(matches),
                search_duration=duration
            )
            db.session.add(history)
            db.session.commit()
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return jsonify({
                'success': True,
                'matches': matches[:Config.MAX_SEARCH_RESULTS],
                'search_duration': duration,
                'search_id': history.id
            })
        
        except Exception as e:
            logger.error(f"Search faces error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search/reverse', methods=['POST'])
    def reverse_image_search():
        """Perform reverse image search on web"""
        try:
            data = request.get_json()
            image_id = data.get('image_id')
            query = data.get('query', '')
            count = data.get('count', 50)
            
            image = ImageRecord.query.get(image_id)
            if not image:
                return jsonify({'error': 'Image not found'}), 404
            
            if not query:
                query = image.original_name
            
            # Run async search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            start = time.time()
            
            results = loop.run_until_complete(
                search_engine.search_multiple_sources(query, count)
            )
            
            loop.close()
            duration = time.time() - start
            
            # Save search history
            history = SearchHistoryRecord(
                search_type='reverse_image',
                image_id=image_id,
                query_params=json.dumps({'query': query, 'count': count}),
                results_count=len(results),
                search_duration=duration
            )
            db.session.add(history)
            db.session.commit()
            
            # Save results
            for result in results[:100]:
                sr = SearchResultRecord(
                    search_id=history.id,
                    image_id=image_id,
                    url=result.get('url'),
                    title=result.get('title'),
                    description=result.get('description'),
                    thumbnail=result.get('thumbnail'),
                    source=result.get('source'),
                    match_score=0.5
                )
                db.session.add(sr)
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'results': results,
                'search_duration': duration,
                'search_id': history.id,
                'count': len(results)
            })
        
        except Exception as e:
            logger.error(f"Reverse search error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/images', methods=['GET'])
    def list_images():
        """List all uploaded images"""
        try:
            images = ImageRecord.query.all()
            return jsonify({
                'success': True,
                'images': [img.to_dict() for img in images],
                'total': len(images)
            })
        except Exception as e:
            logger.error(f"List images error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/image/<int:image_id>', methods=['GET'])
    def get_image_details(image_id):
        """Get image details"""
        try:
            image = ImageRecord.query.get(image_id)
            if not image:
                return jsonify({'error': 'Image not found'}), 404
            
            return jsonify({
                'success': True,
                'image': image.to_dict(),
                'face_count': image.face_count,
                'stats': face_engine.get_face_statistics(image.filepath)
            })
        except Exception as e:
            logger.error(f"Get image error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        """Get system statistics"""
        try:
            total_images = ImageRecord.query.count()
            total_searches = SearchHistoryRecord.query.count()
            total_results = SearchResultRecord.query.count()
            total_faces = sum([img.face_count for img in ImageRecord.query.all()])
            
            avg_duration = db.session.query(
                db.func.avg(SearchHistoryRecord.search_duration)
            ).scalar() or 0
            
            return jsonify({
                'success': True,
                'statistics': {
                    'total_images': total_images,
                    'total_faces': total_faces,
                    'total_searches': total_searches,
                    'total_results': total_results,
                    'avg_search_duration': float(avg_duration)
                }
            })
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app = create_app()
    app.run(
        debug=os.getenv('FLASK_ENV') == 'development',
        host='0.0.0.0',
        port=5000
    )
