from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from io import BytesIO
from PIL import Image

from tensorflow import Graph
import numpy as np

classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

img_height,img_width=256,256

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/2')

# Create your views here.

def home (request):
	return render(request, 'index.html')

def prediction (request):
	try:
		for filename in request.FILES:
			fileObj = request.FILES[filename]
			fs=FileSystemStorage()
			filePathName=fs.save(fileObj.name,fileObj)
			filePathName=fs.url(filePathName)

		testimage='.'+filePathName
		# img = image.load_img(testimage, target_size=(img_height, img_width))
		img = Image.open(testimage)
		img = img.resize((img_width,img_height))
		image = np.array(img)
		img_batch = np.expand_dims(image, 0)
		with model_graph.as_default():
			with tf_session.as_default():
				predi=model.predict(img_batch)

		predictedLabel=classes[np.argmax(predi[0])]
		# print(predi[0])
		# print(np.argmax(predi))
		predPercentage=(predi[0][np.argmax(predi[0])])*100;
		predPercentage = round(predPercentage, 2)

		context = {"filePathName":filePathName, 'predictedLabel':predictedLabel, 'predPercentage':predPercentage}
		error='No'
	except :
		error='yes'
	return render(request, 'index.html',locals())

