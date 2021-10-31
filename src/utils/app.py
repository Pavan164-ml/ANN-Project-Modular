from flask import Flask, render_template,request,jsonify
import pickle
import h5py
import numpy as np

app =Flask(__name__)

# loading the trained model

model = 



@app.route('/',methods=['POST'])
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

    prediction = model.predict(final_features) # making prediction


    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction)) # rendering the predicted result


if __name__ == '__main__':
    app.run(debug=True)