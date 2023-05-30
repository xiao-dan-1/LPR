import tensorflow as tf

print(tf.config.list_physical_devices('GPU'), type(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('CPU'), type(tf.config.list_physical_devices('CPU')))
print(tf.test.is_gpu_available())
