import tensorflow as tf
from keras.models import load_model

def cargarModelo():

    FILENAME_MODEL_TO_LOAD = "covid19_model_full.h5"
    MODEL_PATH = "../../model"

    #cargando el modelo de IA
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("Modelo cargado de disco << ", loaded_model)

    graph = tf.get_default_graph()
    return loaded_model, graph
