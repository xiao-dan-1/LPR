import os
import tensorflow as tf

"""
TF_CPP_MIN_LOG_LEVEL        base_loging	    屏蔽信息	                    输出信息
        “0”	                INFO	        无	                        INFO + WARNING + ERROR + FATAL
        “1”	                WARNING	        INFO	                    WARNING + ERROR + FATAL
        “2”	                ERROR	        INFO + WARNING	            ERROR + FATAL
        “3”	                FATAL	        INFO + WARNING + ERROR	    FATAL
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.debugging.set_log_device_placement(True)

print(tf.config.list_physical_devices('GPU'), "Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('CPU'), "Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# Create some tensors
print("默认使用GPU计算".center(80, '-'))
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

# Place tensors on the CPU
print("使用CPU计算".center(80, '-'))
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
print(c)

print("限制GPU内存增长".center(80, '-'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 限制TensorFlow仅使用第一个GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
