import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
device = torch.device('cpu')
model = models.resnet18(num_classes=2)
model.load_state_dict(torch.load('yorubamodel.pth',map_location=device))
model.eval()



@app.route('/')
def hello():
    return "Hello World!"


def transform_image(infile):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(infile).convert('RGB')
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx
    # return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)            
            return jsonify({'class_id': 0, 'class_name': prediction_idx})

    
if __name__ == '__main__':
    app.run(debug=False)
