from src.utils.common import read_config
from src.utils.model import create_model,save_model
from src.utils.preprocess import preprocessor
from src.utils.data_mgt import get_data,splitting_data
from src.utils.call_back import get_callbacks
import argparse
import os
import logging


def training(config_path):
    try:
        config = read_config(config_path)
        validation_datasize = config["params"]["validation_datasize"]
    except:
        logging.error("Error in reading config file")
        exit(1)
    ## Getting the data

    try:
        data = get_data(validation_datasize)
    except:
        logging.error("Error in getting data")
        exit(1)

    try:
    ## Preprocessing steps
        data = preprocessor(data)
    except:
        logging.error("Error in preprocessing,please check")
        exit(1)

    ## Splittng the data into train valid and test samples
    try:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splitting_data(data,validation_datasize)
    except:
        logging.error("Error in splitting data")
        exit(1)
    
    try:
        LOSS_FUNCTION = config["params"]["loss_function"]
        OPTIMIZER = config["params"]["optimizer"]
        METRICS = config["params"]["metrics"]
        NUM_CLASSES = config["params"]["num_classes"]
    except:
        logging.error("Error in reading parameters from configuration file")
        exit(1)

    try:    
        model = create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES)
    except:
        logging.error("Error in creating model")
        exit(1)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)


    try:
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET)
    except:
        logging.error("Error in training the model")
        exit(1)
                    
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    
    try:
        model_dir_path = os.path.join(artifacts_dir, model_dir)
        os.makedirs(model_dir_path, exist_ok=True)
    except:
        logging.error("Error in creating model directory")

       

    try:
        model_name = config["artifacts"]["model_name"]
        save_model(model, model_name, model_dir_path)
    except:
        logging.error("Error in saving model")
        exit(1)

    logging.warning("Model trained successfully !!!!!")

if __name__ == '__main__':

    logging.basicConfig(filename='training.log', filemode='a', format='%(asctime)s %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',)
    logging.warning("*****************************************************************************************************")
    args = argparse.ArgumentParser("This is just an ArgumentParser")

    args.add_argument("--config","-c",default="config.yml")


    parsed_args = args.parse_args()
    try:
        logging.warning("Training started")
        training(config_path = parsed_args.config)
    except:
        logging.exception("Error in training")
        