from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch.nn as nn
from keras.models import load_model
from PIL import Image
import json

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)

 
#*** Backend operation
 
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads','Fundus_images')
segmented_folder = os.path.join('staticFiles', 'uploads', 'Segmented_images')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

H = 512
W = 512
size = (W, H)
checkpoint_path = os.path.join('models','checkpoint.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

model_2 = load_model(os.path.join('models','my_final.h5'))
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

 
@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Upload file flask
        uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path']= os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def segmentImage():
    """ Reading image """
    img_filename = 'result.jpg'
    image = cv2.imread(session.get('uploaded_img_file_path', None) , cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.resize(image, size)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)
    pred_y = mask_parse(pred_y)
    pred_y = pred_y * 255
    segment_path = os.path.join(segmented_folder, img_filename)
    cv2.imwrite(segment_path, pred_y)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image = segment_path)
@app.route('/detection')
def cataract_detection():

  '''Segmentation'''
  img_filename = 'result.jpg'
  image = cv2.imread(session.get('uploaded_img_file_path', None) , cv2.IMREAD_COLOR) ## (512, 512, 3)
  image = cv2.resize(image, size)
  x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
  x = x/255.0
  x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
  x = x.astype(np.float32)
  x = torch.from_numpy(x)
  x = x.to(device)
  with torch.no_grad():
      pred_y = model(x)
      pred_y = torch.sigmoid(pred_y)
      pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
      pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
      pred_y = pred_y > 0.5
      pred_y = np.array(pred_y, dtype=np.uint8)
  pred_y = mask_parse(pred_y)
  pred_y = pred_y * 255
  segment_path = os.path.join(segmented_folder, img_filename)
  cv2.imwrite(segment_path, pred_y)

  im = Image.open(segment_path)
  im = im.resize((150,150))
  test = np.array(im)
  test = np.expand_dims(test, axis=0)
  prediction = model_2.predict(test)
  predictions = prediction.tolist()[0]
  prediction = np.argmax(predictions)
  if prediction==0:
    output='Detected'
  else:
    output='Not Detected'
  print(prediction)
  percentage = predictions[prediction]
  print(percentage)
  print(predictions)
  return render_template('detection.html', output=output)


 
if __name__=='__main__':
    app.run(debug = True)
