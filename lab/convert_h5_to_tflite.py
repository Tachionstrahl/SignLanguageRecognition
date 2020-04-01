import tensorflow as tf
new_model= tf.keras.models.load_model(filepath="sign_lang_recognition.h5")

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_converter.experimental_new_converter = True
tflite_converter.allow_custom_ops = True

tflite_model = tflite_converter.convert()

open("tf_lite_model.tflite2", "wb").write(tflite_model)