import theano as th
import theano.tensor as T

num_blocks = 3 #Dilated Blocks
num_dim = 128

def get_logit(x, voca_size):

    #residual Block
    def res_block(tensor, size, rate, block, dim = num_dim):
