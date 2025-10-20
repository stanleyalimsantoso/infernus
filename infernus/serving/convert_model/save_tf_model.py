import sys

#from tensorflow.keras.activations import linear
#from tensorflow.keras.models import load_model

import keras
import os

if __name__ == "__main__":


    # Load a saved TensorFlow model and do any 
    # modifications that are required
    #model_path = str(sys.argv[1])
    new_model_path = str(sys.argv[1])
    jobdir = str(sys.argv[2])
    #metamodel_dir = str(sys.argv[3])
    idx = str(sys.argv[3])
    #metamodel_dir = os.path.join(jobdir, metamodel_dir)
    #print("Metamodel dir:", metamodel_dir)
    print("best models are stored in:", os.path.dirname(jobdir))
    #TODO: change to proper metamodel paths after testing

    #old style path below
    #model = keras.models.load_model(os.path.dirname(jobdir)+"/best"+str(idx)+".h5", compile = False)
    model = keras.models.load_model(os.path.join(jobdir,"best"+str(idx)+".h5"), compile = False)
    #model = keras.models.load_model(model_path, compile = False)

    #also save the entire unmodified model
    #TODO: account for models with different layers at the output. 
    #The below code assumes the output is Dense -> lambda -> sigmoid
    model.layers[-3]._name = 'full_out'
    full_model = keras.models.Model(inputs = model.input, outputs = model.layers[-3].output)
    full_model.layers[-1].activation = keras.activations.linear
    #change output layer name to 'output'
    
    print("Full model output name:",full_model.layers[-1].name)

    full_model.compile()
    full_model.save(new_model_path + "_full")