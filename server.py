from flask import Flask, request, jsonify, render_template_string
import base64
import cv2
import numpy as np
from datetime import datetime
import os
import face_recognition

app = Flask(__name__)

# Get port from environment
PORT = int(os.environ.get('PORT', 5000))

# Store recent results in memory
recent_results = []
MAX_RESULTS = 50

# Store known faces
known_face_encodings = []
known_face_names = []

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def load_known_faces():
    """Load known faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    
    known_face_encodings = []
    known_face_names = []
    
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"  ✓ Loaded face: {name}")
    
    print(f"Total known faces loaded: {len(known_face_names)}")

# [Keep all your HTML_TEMPLATE code exactly as is]
HTML_TEMPLATE = """
[Your existing HTML code - no changes needed]
"""

def identify_image(image):
    """Identify faces in the image"""
    # Resize for faster processing on cloud servers
    max_size = 800
    height, width = image.shape[:2]
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use 'hog' model (faster than 'cnn')
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    confidence = 0.0
    
    for face_encoding in face_encodings:
        name = "Unknown"
        face_confidence = 0.0
        
        if len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_confidence = 1 - face_distances[best_match_index]
        
        face_names.append(name)
        confidence = max(confidence, face_confidence)
    
    if len(face_locations) == 0:
        result = "No faces detected"
        confidence = 0.0
    elif len(face_locations) == 1:
        if face_names[0] == "Unknown":
            result = "1 unknown face detected"
        else:
            result = f"Recognized: {face_names[0]}"
    else:
        known_count = sum(1 for name in face_names if name != "Unknown")
        unknown_count = len(face_names) - known_count
        result = f"{len(face_locations)} faces: {known_count} known, {unknown_count} unknown"
    
    height, width = image.shape[:2]
    
    return {
        "result": result,
        "confidence": confidence if confidence > 0 else 0.5,
        "image_size": f"{width}x{height}",
        "timestamp": datetime.now().isoformat(),
        "face_count": len(face_locations),
        "faces": face_names
    }

# [Keep all your route handlers exactly as they are]
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/identify', methods=['POST'])
def identify():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        img_base64 = data['image']
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
        result = identify_image(image)
        
        result_entry = {
            "result": result["result"],
            "confidence": result["confidence"],
            "image_size": result["image_size"],
            "timestamp": result["timestamp"],
            "face_count": result["face_count"],
            "faces": result["faces"],
            "image": img_base64
        }
        
        recent_results.insert(0, result_entry)
        if len(recent_results) > MAX_RESULTS:
            recent_results.pop()
        
        response = {
            "status": "success",
            "result": result["result"],
            "confidence": result["confidence"],
            "image_size": result["image_size"],
            "timestamp": result["timestamp"],
            "face_count": result["face_count"],
            "faces": result["faces"]
        }
        
        print(f"Processed image: {result['result']}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        data = request.get_json()
        
        if 'image' not in data or 'name' not in data:
            return jsonify({"status": "error", "message": "Image and name required"}), 400
        
        name = data['name'].strip()
        img_base64 = data['image']
        
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if len(face_encodings) == 0:
            return jsonify({"status": "error", "message": "No face detected in image"}), 400
        
        if len(face_encodings) > 1:
            return jsonify({"status": "error", "message": "Multiple faces detected. Please use image with single face"}), 400
        
        filename = f"{name}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        cv2.imwrite(filepath, image)
        
        load_known_faces()
        
        print(f"Added new face: {name}")
        return jsonify({"status": "success", "message": f"Face '{name}' added successfully"}), 200
        
    except Exception as e:
        print(f"Error adding face: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        total = len(recent_results)
        total_faces = sum(r.get('face_count', 0) for r in recent_results)
        
        known_faces_list = []
        for name in known_face_names:
            img_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    known_faces_list.append({
                        "name": name,
                        "image": f"data:image/jpeg;base64,{img_data}"
                    })
        
        return jsonify({
            "status": "success",
            "total": total,
            "total_faces_detected": total_faces,
            "known_faces_count": len(known_face_names),
            "known_faces": known_faces_list,
            "results": recent_results[:20]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "known_faces": len(known_face_names)}), 200

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Face Recognition Server")
    print("=" * 60)
    
    load_known_faces()
    
    print("=" * 60)
    print(f"Server starting on port {PORT}")
    print("=" * 60)
    
    # Use PORT from environment for cloud deployment
    app.run(host='0.0.0.0', port=PORT, debug=False)