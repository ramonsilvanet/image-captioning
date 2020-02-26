import os
import shutil
import tensorflow as tf

# Download caption annotation files
annotation_folder = '/datasets/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('./datasets/captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
  annotation_file = os.path.dirname(annotation_zip)+'/datasets/annotations/captions_train2014.json'
  os.remove(annotation_zip)
  os.rename("annotations/", os.path.abspath('.') + annotation_folder)

# Download image files
image_folder = '/datasets/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder

os.rename("train2014/", PATH)


print('Done.')
