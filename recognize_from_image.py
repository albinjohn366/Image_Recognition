from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import sys


def recognize(file, image_1):
    # Loading model
    model = tf.keras.models.load_model(file)

    # Loading image
    image = Image.open(image_1)
    greyscale_image = image.convert('L')
    reshaped_image = greyscale_image.resize((28, 28), Image.ANTIALIAS).filter(
        ImageFilter.SHARPEN)
    image.save('image.png', 'PNG')
    greyscale_image.save('greyscale_image.png', 'PNG')
    reshaped_image.save('reshaped_image.png', 'PNG')

    # getting data
    data = list(reshaped_image.getdata())
    data = [(255 - x) / 255 for x in data]
    data = np.array(data).reshape(1, 28, 28, 1)
    number = model.predict(data).argmax()
    return number


print(recognize(sys.argv[1], sys.argv[2]))
