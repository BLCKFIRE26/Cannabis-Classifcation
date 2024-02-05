from flask import Flask, send_from_directory, request, jsonify

from PIL import Image
import io
import os
import torch
from torchvision import transforms
from main import CNN  

app = Flask(__name__, static_folder='build')


model = CNN()
model.load_state_dict(torch.load('sativa_indica_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return "Welcome to the Flask App"

@app.route('/test')
def test():
    return "Test route working"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file:
            image = Image.open(io.BytesIO(file.read()))
            
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Adjust if necessary
            ])
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                result = model(image)
                prediction = 'Cannabis Sativa' if result[0][0] > 0.5 else 'Cannabis Indica'

            return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': 'Invalid file'}), 500
    pass

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
