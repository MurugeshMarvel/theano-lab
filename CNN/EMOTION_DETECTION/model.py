import theano as th
from theano import tensor as T
from theano import function
from theano import shared
import numpy as np
import utils
import cPickle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

rng = np.random


def load_data():
	with open("emotions-1.pkl",'rb') as file:
		data = cPickle.load(file)
	def make_shared(data_xy, borrow = True):
		data_x, data_y = data_xy
		data_y = (np.asarray(data_y)).T
		print np.shape(data_x)
		print np.shape(data_y)
		shared_x = th.shared(np.asarray(data_x, dtype=th.config.floatX),
								borrow = borrow)
		shared_y = th.shared(np.asarray(data_y, dtype=th.config.floatX),
								borrow = borrow)
		return shared_x, T.cast(shared_y, 'int32')
	train_x, train_y = make_shared(data)
	rval = (train_x, train_y)
	return rval


class emonets(object):
	def __init__(self, rng, input ,image_shape, filter_shape, poolsize=(2,2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input
		fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
		           np.prod(poolsize))
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = th.shared(
		    np.asarray(
		        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
		        dtype=th.config.floatX
		    ),
		    borrow=True
		)
		b_values = np.zeros((filter_shape[0],), dtype=th.config.floatX)
		self.b = th.shared(value=b_values, borrow=True)

		conv_out = conv2d(
		    input=input,
		    filters=self.W,
		    filter_shape=filter_shape,
		    input_shape=image_shape
		)

		pooled_out = pool.pool_2d(
		    input=conv_out,
		    ws=poolsize,
		    ignore_border=True
		)
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		self.params = [self.W, self.b]
		self.input = input
def main(learning_rate=0.1, n_epochs=200,nkerns = [20,50], batch_size = 20):
	rng = np.random.RandomState(23455)
	datasets = load_data()
	train_set_x, train_set_y = datasets
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	index = T.lscalar()
	#th.config.compute_test_value = 'warn'
	x = T.matrix('x')
	y = T.ivector('y')
	print (".../ Building Model")
	layer0_input = x.reshape((batch_size, 1, 350, 350))
	layer0 = emonets(rng, input=layer0_input,
					image_shape = [batch_size,1,350,350],
					filter_shape=(nkerns[0],1,75,75),
					poolsize=(2,2))
	layer1 = emonets(rng, input= layer0.output,
					image_shape = (batch_size, nkerns[0], 138,138),
					filter_shape = (nkerns[1], nkerns[0],75,75),
					poolsize=(2,2))
	layer2_input = layer1.output.flatten(2)
	layer2 = utils.hidden_layer(rng, input = layer2_input,
								n_in = nkerns[1] * 32*32,
								n_out = 500
,								activation =T.tanh)
	layer3 = utils.logistic_regression(input = layer2.output, n_in = 500, n_out = 3)
	cost = layer3.negative_log_likelihood(y)
	params = layer3.params + layer2.params + layer1.params + layer0.params
	grads = T.grad(cost, params)
	updates = [(param_i, param_i - learning_rate * grad_i)
				for param_i, grad_i in zip(params, grads)]

	print train_set_x
	print train_set_y
	train_model = th.function(
					[index],
					cost,
					updates = updates,
					givens={
						x: train_set_x[index * batch_size: (index + 1) * batch_size],
						y: train_set_y[index * batch_size: (index +1) * batch_size]
					})
	print "training"
	epoch = 0
	#done_looping = False
	while (epoch < n_epochs):
		epoch = epoch + 1
		for minibatch_index in range(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index
			if iter % 100 ==0:
				print ("training Iteration ",iter)
			cost_ij = train_model(minibatch_index)
			print ("Cost is ", cost_ij)

if __name__ == '__main__':
	main()
