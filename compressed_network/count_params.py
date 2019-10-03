import sys
import tensorflow as tf
import numpy as np

ckpt_path = '/media/oanaucs/Data/vanilla_pruned/model.ckpt-2000'
reader = tf.train.NewCheckpointReader(ckpt_path)

print('\nCount the number of parameters in ckpt file(%s)' % ckpt_path)
param_map = reader.get_variable_to_shape_map()

# total_count = 0
# for k, v in param_map.items():
#     if 'Momentum' not in k and 'global_step' not in k:
#         temp = np.prod(v)
#         total_count += temp
#         print('%s: %s => %d' % (k, str(v), temp))

# print('Total Param Count: %d' % total_count)

non_zero_count = 0

for key in param_map:
    print("tensor_name: ", key)
    tensor = reader.get_tensor(key)
    non_zero_count += np.count_nonzero(tensor)

print('non zero', non_zero_count)