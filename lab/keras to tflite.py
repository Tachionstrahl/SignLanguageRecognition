import tensorflow as tf
import tensorflow.keras as K
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def convert(input, output):
    np.set_printoptions(threshold=sys.maxsize)

    print("Your tensorflow version:", tf.__version__, "Required >= 2.1.0")

    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 



    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    new_model= tf.keras.models.load_model(filepath=input)

    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    # Needed for some ops.
    tflite_converter.experimental_new_converter = True
    # tflite_converter.allow_custom_ops = True

    tflite_model = tflite_converter.convert()

    open(output, "wb").write(tflite_model)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python keras\ to\ tflite.py path/to/inputfile path/to/outputfile")
        sys.exit(1)
    input = sys.argv[1]
    output = sys.argv[2]
    convert(input, output)