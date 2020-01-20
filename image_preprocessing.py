import os
import logging
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from dotenv import load_dotenv

# load configuration file
load_dotenv()

def config_logger():
  LOGGING_FILE = os.path.join(os.getenv("LOGGING_DIR"), "image_preprocessing.log")
  logging.basicConfig(level=logging.DEBUG,filename=LOGGING_FILE, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

###########################################
#               FUNCTIONS                 #
###########################################

def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	logging.debug(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		logging.debug('>%s' % name)
	return features


###########################################
#             START EXECUTION             #
###########################################
config_logger()

logging.info("Start execution...")

# extract features from all images
DATA_DIR = os.getenv("DATASET_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
IMAGE_DIR = os.path.join(DATA_DIR, os.getenv("IMAGES_DIR"))
VECTORS_DIR = os.path.join(OUTPUT_DIR, os.getenv("VECTORS_DIR"))
FEATURES_FILE = os.path.join(VECTORS_DIR, os.getenv("FEATURES_FILE"))

logging.debug("DATA_DIR", DATA_DIR)
logging.debug("OUTPUT_DIR", OUTPUT_DIR)
logging.debug("IMAGE_DIR", IMAGE_DIR)
logging.debug("VECTORS_DIR", VECTORS_DIR)
logging.debug("FEATURES_FILE", FEATURES_FILE)

features = extract_features(IMAGE_DIR)
logging.info('Extracted Features: %d' % len(features))
# save to file
dump(features, open(FEATURES_FILE, 'wb'))

logging.info("Done.")
