import numpy as np
import theano as th
import cPickle
from theano import tensor as T
with open("emotions-1.pkl",'rb') as file:
    data = cPickle.load(file)
def make_shared(data_xy, borrow = True):
    data_x, data_y = data_xy
    print data_y
    shared_x = th.shared(np.asarray(data_x, dtype=th.config.floatX),
                            borrow = borrow)
    shared_y = th.shared(np.asarray(data_y, dtype=th.config.floatX),
                            borrow = borrow)
    return shared_x, T.cast(shared_y, 'int32')
train_x, train_y = make_shared(data)
print train_x.get_value()
print "#"*300
print train_y.get_value()
