import tensorflow as tf

def increase_sparsity_level(current_value, max_sparsity_level, sparsity_increase_step):
    
    if (current_value < max_sparsity_level):
        return current_value + sparsity_increase_step
    else:
        return max_sparsity_level