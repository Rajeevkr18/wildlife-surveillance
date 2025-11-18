import tensorflow as tf

model = tf.keras.models.load_model("models/animal_classifier.keras")

# Print class order if stored in the model
try:
    print("Class names stored in model:")
    print(model.class_names)
except:
    print("Model does not store class_names attribute.")

# Print model input/output shapes
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
