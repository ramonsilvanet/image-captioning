import os
import logging

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

from dotenv import load_dotenv


# load configuration file
load_dotenv()

def config_logger():
  LOGGING_FILE = os.path.join(os.getenv("LOGGING_DIR"), "train.log")
  logging.basicConfig(level=logging.DEBUG,filename=LOGGING_FILE, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
  console = logging.StreamHandler()
  console.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

###########################################
#               FUNCTIONS                 #
###########################################
# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# load a pre-defined list of photo identifiers
def load_set(filename):
  doc = load_doc(filename)
  dataset = list()
  # process line by line
  for line in doc.split('\n'):
    # skip empty lines
    if len(line) < 1:
      continue
    # get the image identifier
    identifier = line.split('.')[0]
    dataset.append(identifier)
  return set(dataset)

def load_clean_descriptions(filename, dataset):
  # load document
  doc = load_doc(filename)
  descriptions = dict()
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    # split id from description
    image_id, image_desc = tokens[0], tokens[1:]
    # skip images not in the set
    if image_id in dataset:
      # create list
      if image_id not in descriptions:
        descriptions[image_id] = list()
      # wrap description in tokens
      desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
      # store
      descriptions[image_id].append(desc)
  return descriptions

# load photo features
def load_photo_features(filename, dataset):
  # load all features
  all_features = load(open(filename, 'rb'))
  # filter features
  features = {k: all_features[k] for k in dataset}
  return features

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
  all_desc = list()
  for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
  return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
  lines = to_lines(descriptions)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
  X1, X2, y = list(), list(), list()
  # walk through each image identifier
  for key, desc_list in descriptions.items():
    # walk through each description for the image
    for desc in desc_list:
      # encode the sequence
      seq = tokenizer.texts_to_sequences([desc])[0]
      # split one sequence into multiple X,y pairs
      for i in range(1, len(seq)):
        # split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        X1.append(photos[key][0])
        X2.append(in_seq)
        y.append(out_seq)
  return array(X1), array(X2), array(y)

# calculate the length of the description with the most words
def max_length(descriptions):
  lines = to_lines(descriptions)
  return max(len(d.split()) for d in lines)

###########################################
#             START EXECUTION             #
###########################################
config_logger()

logging.info("Start execution...")

DATA_DIR = os.getenv("DATASET_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
IMAGE_DIR = os.path.join(DATA_DIR, os.getenv("IMAGES_DIR"))
VECTORS_DIR = os.path.join(OUTPUT_DIR, os.getenv("VECTORS_DIR"))
FEATURES_FILE = os.path.join(VECTORS_DIR, os.getenv("FEATURES_FILE"))
DESCRIPTIONS_DIR = os.path.join(OUTPUT_DIR, os.getenv("DESCRIPTIONS_DIR"))
DESCRIPTIONS_FILE = os.path.join(DESCRIPTIONS_DIR, os.getenv("DESCRIPTIONS_FILE"))
LABELS_DIR = os.path.join(DATA_DIR, os.getenv("LABELS_DIR"))

logging.debug("DATA_DIR %s" % DATA_DIR)
logging.debug("OUTPUT_DIR %s" % OUTPUT_DIR)
logging.debug("IMAGE_DIR %s" % IMAGE_DIR)
logging.debug("VECTORS_DIR %s" % VECTORS_DIR)
logging.debug("FEATURES_FILE %s" % FEATURES_FILE)
logging.debug("DESCRIPTIONS_FILE %s" % DESCRIPTIONS_FILE)
logging.debug("LABELS_DIR %s" % LABELS_DIR)


# load training dataset (6K)
logging.debug("Loading training dataset")
train_dataset_filename = os.path.join(LABELS_DIR, os.getenv('TRAIN_DATASET_FILENAME'))
train_dataset = load_set(train_dataset_filename)
logging.info('Dataset: %d' % len(train_dataset))

# descriptions
logging.debug("Loading training descriptions")
train_descriptions = load_clean_descriptions(DESCRIPTIONS_FILE, train_dataset)
logging.info('Descriptions dataset: train=%d' % len(train_descriptions))
# photo features
logging.debug("Loading training photo features vectors")
train_features = load_photo_features(FEATURES_FILE, train_dataset)
logging.info('Photos features vectors: train=%d' % len(train_features))

# prepare tokenizer
logging.info("tokenizing words")
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
logging.info('Vocabulary Size: %d' % vocab_size)

logging.info("Staring encode text data")

logging.info("Done.")
