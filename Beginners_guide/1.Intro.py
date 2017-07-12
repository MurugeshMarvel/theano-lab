import theano
print "Intro to Theano"
############################
print "#"*30
print "Scalar Addition"
a = theano.tensor.scalar('a')
b = theano.tensor.scalar('b')
c = a + b
scalar_add_function = theano.function([a,b],[c])
print scalar_add_function(4,6)
############################
print '#'*30
print "Vector Addition"
a = theano.tensor.vector('a')
b = theano.tensor.vector('b')
c = a + b
vector_add_function = theano.function([a,b],[c])
print vector_add_function([2,3,4],[3,2,7])
############################
print '#'*30
print "Matrix Addition"
a = theano.tensor.matrix('a')
b = theano.tensor.matrix('b')
c = a + b
matrix_add_function = theano.function([a,b],[c])
print matrix_add_function(((2,3,4),(3,2,7),(2,6,4)),((5,2,4),(6,2,6),(9,7,1)))

print "Using Eval Method"
print c.eval({a : ((2,3,4),(3,2,7),(2,6,4)), b :((5,2,4),(6,2,6),(9,7,1))})