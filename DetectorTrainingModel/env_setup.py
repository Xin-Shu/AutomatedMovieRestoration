import os
import tensorflow.compat.v1 as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['DML_VISIBLE_DEVICES'] = '0'
tf.debugging.set_log_device_placement(True)
tf.enable_eager_execution()

tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 
print(tf.add([1.0, 2.0], [3.0, 4.0])) 