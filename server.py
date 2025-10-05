from flask import Flask, request, jsonify, render_template_string
import base64
import cv2
import numpy as np
from datetime import datetime
import json
import os
import face_recognition

app = Flask(__name__)

# Store recent results in memory (in production, use a database)
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
                # Use filename without extension as person's name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"  ✓ Loaded face: {name}")
    
    print(f"Total known faces loaded: {len(known_face_names)}")

# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        #latestImage {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .result-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .result-info p {
            margin: 8px 0;
            color: #555;
        }
        
        .result-info strong {
            color: #333;
        }
        
        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .results-table {
            width: 100%;
            max-height: 500px;
            overflow-y: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        .status-active {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-inactive {
            color: #dc3545;
        }
        
        .upload-section {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        
        .upload-area input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            margin-top: 15px;
            margin-right: 10px;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .upload-btn.secondary {
            background: linear-gradient(135deg, #f093fb, #f5576c);
        }
        
        #uploadPreview {
            max-width: 300px;
            margin: 20px auto;
            border-radius: 10px;
            display: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .face-badge {
            display: inline-block;
            padding: 5px 12px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            margin: 5px;
            font-size: 0.9em;
        }
        
        .face-badge.unknown {
            background: #dc3545;
        }
        
        .known-faces-section {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
        }
        
        .face-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .face-item {
            text-align: center;
        }
        
        .face-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        
        .face-item p {
            margin-top: 8px;
            font-weight: bold;
            color: #333;
        }
        
        .name-input {
            padding: 10px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 1em;
            margin-top: 15px;
            width: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>👤 Face Recognition System</h1>
            <p class="subtitle">Real-time Face Detection & Recognition Dashboard</p>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number" id="totalImages">0</div>
                <div class="stat-label">Total Images Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="knownFaces">0</div>
                <div class="stat-label">Known Faces</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="facesDetected">0</div>
                <div class="stat-label">Total Faces Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="systemStatus" class="status-active">Active</div>
                <div class="stat-label">System Status</div>
            </div>
        </div>
        
        <div class="known-faces-section">
            <h2>📚 Known Faces Database</h2>
            <p style="color: #666; margin-bottom: 15px;">Upload images to the <strong>known_faces/</strong> folder with format: <code>PersonName.jpg</code></p>
            <div class="face-grid" id="knownFacesGrid">
                <p style="color: #999; grid-column: 1/-1; text-align: center;">No known faces yet. Add images to known_faces/ folder.</p>
            </div>
        </div>
        
        <div class="upload-section">
            <h2>📤 Upload Image for Face Recognition</h2>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>Click to upload an image or drag and drop</p>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            </div>
            <img id="uploadPreview" alt="Preview">
            <div class="loading" id="uploadLoading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
            <div id="uploadActions" style="display:none; text-align: center;">
                <button class="upload-btn" onclick="recognizeFace()">🔍 Recognize Face</button>
                <button class="upload-btn secondary" onclick="showAddFaceDialog()">➕ Add to Known Faces</button>
            </div>
            <div id="addFaceDialog" style="display:none; text-align: center; margin-top: 20px;">
                <input type="text" id="personName" class="name-input" placeholder="Enter person's name">
                <br>
                <button class="upload-btn" onclick="addKnownFace()">Save Face</button>
                <button class="upload-btn secondary" onclick="cancelAddFace()">Cancel</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>📸 Latest Detection</h2>
                <img id="latestImage" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300'%3E%3Crect width='400' height='300' fill='%23f0f0f0'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23999' font-size='20'%3EWaiting for image...%3C/text%3E%3C/svg%3E" alt="Latest detection">
                <div class="result-info">
                    <p><strong>Result:</strong> <span id="latestResult">-</span></p>
                    <p><strong>Faces Detected:</strong> <span id="latestFaceCount">0</span></p>
                    <div id="latestFaces"></div>
                    <p><strong>Confidence:</strong></p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="latestConfidence" style="width: 0%">0%</div>
                    </div>
                    <p><strong>Timestamp:</strong> <span id="latestTime">-</span></p>
                    <p><strong>Image Size:</strong> <span id="latestSize">-</span></p>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Recent Results</h2>
                <div class="results-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Result</th>
                                <th>Faces</th>
                            </tr>
                        </thead>
                        <tbody id="resultsBody">
                            <tr>
                                <td colspan="3" style="text-align: center; color: #999;">No results yet</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let uploadedImage = null;
        
        // Fetch and update dashboard data
        function updateDashboard() {
            fetch('/api/results')
                .then(response => response.json())
                .then(data => {
                    if (data.results && data.results.length > 0) {
                        // Update stats
                        document.getElementById('totalImages').textContent = data.total;
                        document.getElementById('knownFaces').textContent = data.known_faces_count || 0;
                        document.getElementById('facesDetected').textContent = data.total_faces_detected || 0;
                        
                        // Update latest result
                        const latest = data.results[0];
                        if (latest.image) {
                            document.getElementById('latestImage').src = 'data:image/jpeg;base64,' + latest.image;
                        }
                        document.getElementById('latestResult').textContent = latest.result;
                        document.getElementById('latestFaceCount').textContent = latest.face_count || 0;
                        
                        // Display face names
                        const facesDiv = document.getElementById('latestFaces');
                        facesDiv.innerHTML = '';
                        if (latest.faces && latest.faces.length > 0) {
                            latest.faces.forEach(face => {
                                const badge = document.createElement('span');
                                badge.className = face === 'Unknown' ? 'face-badge unknown' : 'face-badge';
                                badge.textContent = face;
                                facesDiv.appendChild(badge);
                            });
                        }
                        
                        const confidence = (latest.confidence * 100).toFixed(1);
                        document.getElementById('latestConfidence').style.width = confidence + '%';
                        document.getElementById('latestConfidence').textContent = confidence + '%';
                        document.getElementById('latestTime').textContent = 
                            new Date(latest.timestamp).toLocaleString();
                        document.getElementById('latestSize').textContent = latest.image_size;
                        
                        // Update table
                        const tbody = document.getElementById('resultsBody');
                        tbody.innerHTML = '';
                        data.results.forEach(item => {
                            const row = tbody.insertRow();
                            row.insertCell(0).textContent = new Date(item.timestamp).toLocaleTimeString();
                            row.insertCell(1).textContent = item.result;
                            row.insertCell(2).textContent = item.face_count || 0;
                        });
                    }
                    
                    // Update known faces grid
                    if (data.known_faces && data.known_faces.length > 0) {
                        const grid = document.getElementById('knownFacesGrid');
                        grid.innerHTML = '';
                        data.known_faces.forEach(face => {
                            const item = document.createElement('div');
                            item.className = 'face-item';
                            item.innerHTML = `
                                <img src="${face.image}" alt="${face.name}">
                                <p>${face.name}</p>
                            `;
                            grid.appendChild(item);
                        });
                    }
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        
        // Handle file selection
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('uploadPreview').src = e.target.result;
                    document.getElementById('uploadPreview').style.display = 'block';
                    document.getElementById('uploadActions').style.display = 'block';
                    uploadedImage = e.target.result.split(',')[1];
                };
                reader.readAsDataURL(file);
            }
        }
        
        // Recognize face
        function recognizeFace() {
            if (!uploadedImage) return;
            
            document.getElementById('uploadLoading').style.display = 'block';
            
            fetch('/identify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: uploadedImage })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('uploadLoading').style.display = 'none';
                
                if (data.status === 'success') {
                    let msg = `Result: ${data.result}\\nFaces: ${data.face_count}`;
                    if (data.faces && data.faces.length > 0) {
                        msg += '\\n\\nDetected:\\n' + data.faces.join('\\n');
                    }
                    alert(msg);
                    updateDashboard();
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                document.getElementById('uploadLoading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
        
        // Show add face dialog
        function showAddFaceDialog() {
            document.getElementById('addFaceDialog').style.display = 'block';
        }
        
        // Cancel add face
        function cancelAddFace() {
            document.getElementById('addFaceDialog').style.display = 'none';
            document.getElementById('personName').value = '';
        }
        
        // Add known face
        function addKnownFace() {
            const name = document.getElementById('personName').value.trim();
            if (!name) {
                alert('Please enter a name');
                return;
            }
            
            if (!uploadedImage) return;
            
            document.getElementById('uploadLoading').style.display = 'block';
            
            fetch('/add_face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    image: uploadedImage,
                    name: name
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('uploadLoading').style.display = 'none';
                
                if (data.status === 'success') {
                    alert('Face added successfully!');
                    cancelAddFace();
                    updateDashboard();
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                document.getElementById('uploadLoading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
        
        // Initial update and set interval
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""

def identify_image(image):
    """
    Identify faces in the image
    """
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    confidence = 0.0
    
    # Loop through each face found
    for face_encoding in face_encodings:
        name = "Unknown"
        face_confidence = 0.0
        
        if len(known_face_encodings) > 0:
            # See if the face matches any known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    face_confidence = 1 - face_distances[best_match_index]
        
        face_names.append(name)
        confidence = max(confidence, face_confidence)
    
    # Generate result message
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

@app.route('/')
def index():
    """Serve the web dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/identify', methods=['POST'])
def identify():
    """Receive image, process it, and return results"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                "status": "error",
                "message": "No image provided"
            }), 400
        
        # Decode base64 image
        img_base64 = data['image']
        img_bytes = base64.b64decode(img_base64)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "status": "error",
                "message": "Failed to decode image"
            }), 400
        
        # Identify faces
        result = identify_image(image)
        
        # Store result with image
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
        
        # Return response
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
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add a new face to known faces"""
    try:
        data = request.get_json()
        
        if 'image' not in data or 'name' not in data:
            return jsonify({
                "status": "error",
                "message": "Image and name required"
            }), 400
        
        name = data['name'].strip()
        img_base64 = data['image']
        
        # Decode image
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if face exists
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if len(face_encodings) == 0:
            return jsonify({
                "status": "error",
                "message": "No face detected in image"
            }), 400
        
        if len(face_encodings) > 1:
            return jsonify({
                "status": "error",
                "message": "Multiple faces detected. Please use image with single face"
            }), 400
        
        # Save image to known_faces directory
        filename = f"{name}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        cv2.imwrite(filepath, image)
        
        # Reload known faces
        load_known_faces()
        
        print(f"Added new face: {name}")
        
        return jsonify({
            "status": "success",
            "message": f"Face '{name}' added successfully"
        }), 200
        
    except Exception as e:
        print(f"Error adding face: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get recent results for dashboard"""
    try:
        total = len(recent_results)
        total_faces = sum(r.get('face_count', 0) for r in recent_results)
        
        # Get known faces with images
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
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "known_faces": len(known_face_names)
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Face Recognition Server")
    print("=" * 60)
    
    # Load known faces on startup
    load_known_faces()
    
    print("=" * 60)
    print(f"Web dashboard: http://0.0.0.0:5000")
    print(f"API Endpoint: http://0.0.0.0:5000/identify")
    print(f"Health Check: http://0.0.0.0:5000/health")
    print(f"Known Faces Directory: {KNOWN_FACES_DIR}/")
    print("=" * 60)
    print("\nTo add known faces:")
    print("1. Place images in 'known_faces/' folder")
    print("2. Name format: PersonName.jpg")
    print("3. Or use the web interface to add faces")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)