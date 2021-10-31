import tensorflow as tf
from tensorflow import keras
import os
import logging
import pickle
import time

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):
    model = keras.Sequential([
        keras.layers.Dense(26,input_shape = (26,), activation = 'relu'),
        keras.layers.Dense(15,activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ])

    model_clf = tf.keras.models.Sequential(model)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)
    
    return model_clf ## <<< untrained model

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)
    
    
    # pickle.dump(model,open('model123.pkl','wb'))
    
    # pickle_out = open("model.pkl","wb")
    # pickle.dump(model, pickle_out)
    # pickle_out.close()


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename
