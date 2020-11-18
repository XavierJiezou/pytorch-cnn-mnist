from flask import Flask, jsonify, request, render_template
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from model import CNN
import base64
import torch
import io
import re
import os


app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


model = CNN()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        output = model.forward(tensor)
        confidence = F.softmax(output, 1).max()*100
        output = output.argmax(1)
        return output.item(), f'{confidence.item():.2f}%'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['POST'])
def predict():
    img = request.get_data()
    img = re.search(b'base64,(.*)', img).group(1)
    img = base64.decodebytes(img)
    out = get_prediction(img)
    return jsonify({'prediction': out[0], 'confidence': out[1]})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')