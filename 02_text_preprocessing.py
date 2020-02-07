import os
import logging
import string
from os import listdir
from dotenv import load_dotenv

# load configuration file
load_dotenv()

###########################################
#               FUNCTIONS                 #
###########################################
def config_logger():
  LOGGING_FILE = os.path.join(os.getenv("LOGGING_DIR"), "text_preprocessing.log")
  logging.basicConfig(level=logging.DEBUG,filename=LOGGING_FILE, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# extract descriptions for images
def load_descriptions(doc):
  mapping = dict()
  # process lines
  for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    if len(line) < 2:
      continue
    # take the first token as the image id, the rest as the description
    image_id, image_desc = tokens[0], tokens[1:]
    # remove filename from image id
    image_id = image_id.split('.')[0]
    # convert description tokens back to string
    image_desc = ' '.join(image_desc)
    # create the list if needed
    if image_id not in mapping:
      mapping[image_id] = list()
    # store description
    mapping[image_id].append(image_desc)
  return mapping

def clean_descriptions(descriptions):
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
      desc = desc_list[i]
      # tokenize
      desc = desc.split()
      # convert to lower case
      desc = [word.lower() for word in desc]
      # remove punctuation from each token
      desc = [w.translate(table) for w in desc]
      # remove hanging 's' and 'a'
      desc = [word for word in desc if len(word)>1]
      # remove tokens with numbers in them
      desc = [word for word in desc if word.isalpha()]
      # store as string
      desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
  # build a list of all description strings
  all_desc = set()
  for key in descriptions.keys():
    [all_desc.update(d.split()) for d in descriptions[key]]
  return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
  lines = list()
  for key, desc_list in descriptions.items():
    for desc in desc_list:
      lines.append(key + ' ' + desc)
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

###########################################
#             START EXECUTION             #
###########################################
config_logger()

logging.info("Start execution...")

DATA_DIR = os.getenv("DATASET_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
LABELS_DIR = os.path.join(DATA_DIR, os.getenv("LABELS_DIR"))
DESCRIPTIONS_DIR = os.path.join(OUTPUT_DIR, os.getenv("DESCRIPTIONS_DIR"))
TRAIN_LABELS_FILE = os.path.join(LABELS_DIR, os.getenv("TRAIN_LABELS_FILE"))
DESCRIPTIONS_FILE = os.path.join(DESCRIPTIONS_DIR, os.getenv("DESCRIPTIONS_FILE"))

logging.debug("DATA_DIR %s" % DATA_DIR)
logging.debug("OUTPUT_DIR %s" % OUTPUT_DIR)
logging.debug("LABELS_DIR %s" % LABELS_DIR)
logging.debug("TRAIN_LABELS_FILE %s" % TRAIN_LABELS_FILE)
logging.debug("DESCRIPTIONS_FILE %s" % DESCRIPTIONS_FILE)

# load descriptions
doc = load_doc(TRAIN_LABELS_FILE)

# parse descriptions
descriptions = load_descriptions(doc)
logging.info('Loaded: %d ' % len(descriptions))

# clean descriptions
clean_descriptions(descriptions)

# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
logging.info('Vocabulary Size: %d' % len(vocabulary))

# save descriptions
logging.info("descriptions saved in %s" % DESCRIPTIONS_FILE)
save_descriptions(descriptions, DESCRIPTIONS_FILE)

logging.info("Done.")
