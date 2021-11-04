from flask import Flask, render_template,request,jsonify
import tensorflow as tf
import numpy as np
import logging

app =Flask(__name__)

FILE_PATH = "D:/Folders/Coding Stuff/Machine learning stuff/iNeuron/Projects/Perfect Implementations/ANN-Project-Modular/artifacts/model/20211031_195453_model.h5"
# loading the trained model
model = tf.keras.models.load_model(FILE_PATH)
print("*"*60)
print(model)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    print(final_features.shape)
    prediction = model.predict(final_features) # making prediction

    return render_template('after.html', data=prediction) # rendering the predicted result


if __name__ == '__main__':
    app.run(debug=True)