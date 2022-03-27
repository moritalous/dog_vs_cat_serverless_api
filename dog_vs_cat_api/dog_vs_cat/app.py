import json

import base64
from requests_toolbelt.multipart import decoder

import tensorflow as tf

import numpy as np

IMG_HEIGHT = 150
IMG_WIDTH = 150

model = tf.keras.models.load_model('dog_vs_cat.h5')

def preprocess_image(image):
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  image /= 255.0  # normalize to [0,1] range

  return image


def get_image(event):
    
    content_type = event['headers']['content-type']
    decode_body = base64.b64decode(event['body'])

    multipart_body = decoder.MultipartDecoder(decode_body, content_type)

    for part in multipart_body.parts:
        try:
            c_type = part.headers.get(b'Content-Type')
            if c_type and c_type == b'image/jpeg':
              return part.content
        except:
            pass


def lambda_handler(event, context):

    print(json.dumps(event))

    img = get_image(event)
    img = preprocess_image(img)
    img = (np.expand_dims(img,0))

    prediction = model.predict(img)

    message = "dog"
    if float(prediction[0,0]) < 0.5:
        message = "cat"

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": message,
                "predict": float(prediction[0,0])
            }
        ),
    }


if __name__ == "__main__":
    pass
