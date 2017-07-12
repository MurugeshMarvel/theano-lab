import theano
import theano.tensor as T
x = T.dmatrix('x')
s = 1/(1 + T.exp(-x))
logistic = theano.function([x],s)
print logistic([[0,1], [-1,-2]])
s2 = (1 + T.tanh(x/2)) / 2
logistic2 = theano.function([x], s2)
logistic2([[0,1],[-1, -2]])