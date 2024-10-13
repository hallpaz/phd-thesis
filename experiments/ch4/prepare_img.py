import os
import numpy as np
from PIL import Image, ImageFilter

DATA_DIR = os.path.join('data', 'img')


img = Image.open(os.path.join(DATA_DIR, 'sloth.jpg'))

grayscale = img.convert('L')
grayscale.save(os.path.join(DATA_DIR, 'grayscale.png'))

edges = img.filter(ImageFilter.FIND_EDGES)
edges.save(os.path.join(DATA_DIR, 'edges.png'))

blurred = grayscale.filter(ImageFilter.GaussianBlur(5))
blurred.save(os.path.join(DATA_DIR, 'blurred.png'))

dog = np.array(grayscale, np.int32) - np.array(blurred, np.int32)
dog = dog - np.min(dog)
dog = 255 * dog / np.max(dog)
dog = Image.fromarray(dog.astype(np.uint8))
dog.save(os.path.join(DATA_DIR, 'dog.png'))