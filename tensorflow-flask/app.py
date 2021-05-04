from flask import Flask
import struct
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize
import time
import math
import numpy as np
import json
import certifi
from elasticsearch import Elasticsearch, helpers, RequestsHttpConnection
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64

es = es = Elasticsearch(hosts=['https://admin:Admin123!!@search-matrixhaki-jstrc22xec3ermwuwloireehsy.us-east-2.es.amazonaws.com'], ca_certs=certifi.where())
def preprocess(img_path, input_shape):
	img = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img, channels=input_shape[2])
	img = tf.image.resize(img, input_shape[:2])
	img = preprocess_input(img)
	return img

batch_size = 100
input_shape = (224,224,3)
base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base.trainable = False
model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

app = Flask(__name__)

@app.route("/testing")
def testing():
	return "hello world ubuntu"

@app.route("/index/<img64>/<img_opt>")
def indexku(img64, img_opt):

	base64_message = img64
	base64_bytes = base64_message.encode('ascii')
	message_bytes = base64.b64decode(base64_bytes)
	message = message_bytes.decode('ascii')

	fnames = [message]
	list_ds = tf.data.Dataset.from_tensor_slices(fnames)
	ds = list_ds.map(lambda x: preprocess(x, input_shape), num_parallel_calls=-1)
	dataset = ds.batch(batch_size).prefetch(-1)
	fvecs = model.predict(dataset)
	
	#fmt = f'{np.prod(fvecs.shape)}f'
	#hoho = fmt, *(fvecs.flatten())
	get_vector = normalize(fvecs)[0].tolist()

	k = 12

	if img_opt == 'merek':
		#idx_name = 'djki_ai2'
		idx_name = 'haki_merk'
	else:
		idx_name = 'design_industri'	

	res = es.search(request_timeout=80, index=idx_name, body={'size': k, '_source': { "includes": ["filename"]}, 'query': { 'knn': { 'fvec': { 'vector': get_vector, 'k': k} }}})

	resulting = json.dumps(res, indent=6)

	return resulting

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)