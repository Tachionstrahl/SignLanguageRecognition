import pandas as pd
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

from keras.models import load_model
model = load_model('LSTM.h5')

model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model2")

inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input)
}
outputs = {
    'earnings': tf.saved_model.utils.build_tensor_info(model.output)
}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()
