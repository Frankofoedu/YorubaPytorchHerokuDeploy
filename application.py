import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import json
import torch
import pathlib
import librosa
import os
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

#pytorch load
device = torch.device('cpu')
model = models.resnet18(num_classes=2)
model.load_state_dict(torch.load('yorubamodel.pth',map_location=device))
model.eval()

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))


@app.route('/')
def hello():
	return "Hello World!"


def transform_image(infile):
	my_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	image = Image.open(infile).convert('RGB')
	timg = my_transforms(image)
	timg.unsqueeze_(0)
	return timg


def get_prediction(input_tensor):
	outputs = model.forward(input_tensor)
	_, y_hat = outputs.max(1)
	predicted_idx = str(y_hat.item())
	toop = torch.nn.functional.softmax(outputs,dim=1)
	v, t =(toop.topk(1)) #gets the probability of the output class	
	v = float("%.2f" % v.data.numpy().item()) * 100 #convert to percentage
	return predicted_idx, v
	# return imagenet_class_index[predicted_idx]


#save file to tmp folder
def get_spectrogram(filename):
	songname = f'tmp/{filename}'
	y, sr = librosa.load(songname, mono=True, duration=5)
	plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
	plt.axis('off')
	plt.savefig(f'tmp/{filename[:-3].replace(".", "")}.png')
	plt.clf()
	return f'tmp/{filename[:-3].replace(".", "")}.png'
	
	
@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		audio_file = request.files['audio_file']
		if audio_file is not None:
			filename = secure_filename(audio_file.filename)
			audio_file.save('tmp/'+ filename)
			imgfile = get_spectrogram(filename)
			input_tensor = transform_image(imgfile)
			prediction_idx = get_prediction(input_tensor)			  
			return jsonify({'class_id': prediction_idx, 'probability' : probability})
		else:
			abort(400)

	
if __name__ == '__main__':
	app.run(debug=False)
