from flask import Flask , render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

#dic = {'0_normal': 0, '1_ulcerative_colitis': 1, '2_polyps': 2, '3_esophagitis': 3}

from tensorflow.keras.layers import TFSMLayer

# Load the model as a TFSMLayer
model_layer = TFSMLayer('my_efficientnet_model', call_endpoint='serving_default')

# To use the model for inference, you can wrap it in a Keras Model
from tensorflow.keras import Input, Model

# Assuming your model expects an input shape, e.g., (224, 224, 3) for EfficientNet
inputs = Input(shape=(224, 224, 3))
outputs = model_layer(inputs)
model = Model(inputs, outputs)

#model.make_predict_function()
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
from tensorflow.keras.preprocessing import image

def predict_label(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    #img_array = preprocess_input(img_array)  # Preprocess the image data

    # Reshape and preprocess the input data
    img_arrayy= np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_arrayy)
    prediction_array = predictions['dense_5']

    # Find the index of the maximum value in the prediction array
    predicted_index = np.argmax(prediction_array, axis=-1)
    # Get the predicted index with maximum probability
    #predicted_index = np.argmax(predictions,-1)

    # Determine the predicted label based on the index
    if predicted_index == 0:
        predicted_label = "Normal"
    elif predicted_index == 1:
        predicted_label = "Ulcerative colitis"
    elif predicted_index == 2:
        predicted_label = "Polyp"
    else:
        predicted_label = "Esophagitis"

    return predicted_label


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index1.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        # Print the image path for debugging
        print("Image Path:", img_path)

        # Get the prediction
        p = predict_label(img_path)

        # Print the prediction for debugging
        print("Prediction:", p)

    return render_template("result.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
	#app.debug = True
	app.run(debug=True)
