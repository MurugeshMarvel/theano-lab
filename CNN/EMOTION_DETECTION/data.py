import numpy as np
import cPickle
import Image
import os
import theano

class data_feature(object):
	def load_images(self,path):
		label = []
		num = 0
		for a,b,c in os.walk(path):
			for i in b:
				label.append(i)
		return label
	def image_feature(self, image_path):
		img = Image.open(image_path)
		img_arr = np.asarray(img,dtype='float32')
		a = img_arr.flatten()
		data = np.asarray([a])
		return data
	def train_data(self):
		'''
		def shared(data_xy, borrow=True):
			data_x, data_y = data_xy
			shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
									borrow = borrow)
			shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
									borrow=borrow)
			return shared_x, T.cast(shared_y, 'int32')
			'''
		label_id = 0
		loop = 0
		y = []
		for i in self.load_images("dataset/"):
			dir = "dataset/" + i +'/'
			for a,b,c in os.walk(dir):
				for i in c:
					data_dir = (str(a) + str(i))
					data = self.image_feature(data_dir)
					if loop == 0:
						#print data
						x= np.asarray(data)
						print np.shape(data)
						print np.shape(x)
						y.append(label_id)
						print "#"*30
						print np.shape(label_id)
						print np.shape(y)
					else:
						#lab_id = np.asarray([[label_id]])
						x = np.concatenate((x,data), axis=0)
						y.append(label_id)
						#y = np.concatenate((y,lab_id), axis=0)
					loop += 1
			label_id  += 1
		print "The shape of the input matrix",np.shape(x)
		print "The shape of the output matrix",np.shape(y)
		return (x,y)
if __name__ == "__main__":
	obj = data_feature()
	train = obj.train_data()
'''
	with open("emotions-1.pkl",'wb') as fi:
		cPickle.dump(train, fi)
	print ("Features have been successfully converted from images and stored in emotions.pkl file")
	print("Next phase")
'''