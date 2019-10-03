import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint
from tensorflow.python import pywrap_tensorflow

def load_variables(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_values = dict()
    for key in sorted(var_to_shape_map):
        # print("tensor_name: ", key)
        # print(reader.get_tensor(key))
        var_values.update({key: reader.get_tensor(key)})
    return var_values



def main():
    ckpt_filename = '../../awp_vgg_10_pruned/model.ckpt-0'
    load_variables(ckpt_filename)

if __name__ == '__main__':
    main()

