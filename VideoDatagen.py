from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from random import Random
import os
import threading

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return next(self.it)

def threadsafe_generator(f):
		"""A decorator that takes a generator function and makes it thread-safe.
		"""
		def g(*a, **kw):
			return threadsafe_iter(f(*a, **kw))
		return g

class VideoDatagen:
	def __init__(self, img_size, target_size, batch_size, random_seed, data_dir="data"):
		self.img_size = img_size
		self.batch_size = batch_size
		self.random_seed = random_seed
		self.target_size = target_size
		self.train_datagen = ImageDataGenerator(rescale=1./255)
		self.val_datagen = ImageDataGenerator(rescale=1./255)
		self.test_datagen = ImageDataGenerator(rescale=1./255)
		self.BS = 3
		self.train_random = Random(self.random_seed)
		self.val_random = Random(self.random_seed)
		self.test_random = Random(self.random_seed)
		self.data_dir = data_dir

		self.train_sharp = self.train_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"train","train_sharp"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)
		
		self.train_blur = self.train_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"train","train_blur"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)

		self.val_sharp = self.val_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"val","val_sharp"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)
		
		self.val_blur = self.val_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"val","val_blur"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)
		
		self.test_sharp = self.test_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"test","test_sharp"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)

		self.test_blur = self.test_datagen.flow_from_directory(
	        os.path.join(self.data_dir,"test","test_blur"),
	        seed=self.random_seed,
	        target_size=self.img_size,
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='sparse',
	        batch_size=self.BS)
		
		self.train_samples = self.train_sharp.samples
		self.val_samples = self.val_sharp.samples
		self.test_samples = self.test_sharp.samples

	def randomCrop(self, imgs, masks, height, width, random):
	    
	    x = random.randint(0, imgs[0].shape[1] - width)
	    y = random.randint(0, imgs[0].shape[0] - height)
	    
	    imgs_out = imgs[:,y:y+height, x:x+width,:]
	    masks_out = masks[:,y:y+height, x:x+width,:]
	    
	    return imgs_out, masks_out

	@threadsafe_generator
	def train_generator(self):			    
	    while True: 
	        i = 0

	        X_out = np.empty((0, self.target_size[0], self.target_size[1], 9))
	        y_out = np.empty((0, self.target_size[0], self.target_size[1], 3))

	        while i  < self.batch_size:       
	            Xi = self.train_blur.next()
	            Xi_imgs = Xi[0]
	            Xi_labels = Xi[1]
	            Yi = self.train_sharp.next()
	            Yi_imgs = Yi[0]
	            Yi_labels = Yi[1]

	            if np.min(Xi_labels) == np.max(Xi_labels):
	                (Xi_imgs, Yi_imgs) = self.randomCrop(Xi_imgs, Yi_imgs, self.target_size[0], self.target_size[1], self.train_random)
					
					#Data Augmentation for the training set
	                direction = self.train_random.randint(1,4)
	                amount = 2

	                if direction is 1:            
	                    M1 = np.float32([[1,0,amount],[0,1,0]])
	                    M2 = np.float32([[1,0,-amount],[0,1,0]])
	                elif direction is 2:
	                    M1 = np.float32([[1,0,-amount],[0,1,0]])
	                    M2 = np.float32([[1,0,amount],[0,1,0]])
	                elif direction is 3:
	                    M1 = np.float32([[1,0,0],[0,1,-amount]])
	                    M2 = np.float32([[1,0,0],[0,1,amount]])
	                elif direction is 4:
	                    M1 = np.float32([[1,0,0],[0,1,amount]])
	                    M2 = np.float32([[1,0,0],[0,1,-amount]])

	                rotation = self.train_random.uniform(-90.0,90.0)
	                M_rotation = cv2.getRotationMatrix2D(((self.target_size[0]-1)/2.0,(self.target_size[1]-1)/2.0),rotation,1)
	                Rot = np.vstack([M_rotation, [0,0,1]])

	                M1 = np.matmul(M1, Rot)
	                M2 = np.matmul(M2, Rot)

	                Xi_imgs[0] = cv2.warpAffine(Xi_imgs[0], M1, self.target_size, borderMode=cv2.BORDER_REPLICATE)
	                Xi_imgs[1] = cv2.warpAffine(Xi_imgs[1], M_rotation, self.target_size, borderMode=cv2.BORDER_REPLICATE)
	                Xi_imgs[2] = cv2.warpAffine(Xi_imgs[2], M2, self.target_size, borderMode=cv2.BORDER_REPLICATE)
	                Yi_imgs[1] = cv2.warpAffine(Yi_imgs[1], M_rotation, self.target_size, borderMode=cv2.BORDER_REPLICATE)

	                direction = self.train_random.randint(1,4)

	                if direction is 1:
	                    Xi_imgs[0] = cv2.flip(Xi_imgs[0], 0)
	                    Xi_imgs[1] = cv2.flip(Xi_imgs[1], 0)
	                    Xi_imgs[2] = cv2.flip(Xi_imgs[2], 0)
	                    Yi_imgs[1] = cv2.flip(Yi_imgs[1], 0)
	                elif direction is 2:
	                    Xi_imgs[0] = cv2.flip(Xi_imgs[0], 1)
	                    Xi_imgs[1] = cv2.flip(Xi_imgs[1], 1)
	                    Xi_imgs[2] = cv2.flip(Xi_imgs[2], 1)
	                    Yi_imgs[1] = cv2.flip(Yi_imgs[1], 1)
	                elif direction is 3:
	                    Xi_imgs[0] = cv2.flip(Xi_imgs[0], -1)
	                    Xi_imgs[1] = cv2.flip(Xi_imgs[1], -1)
	                    Xi_imgs[2] = cv2.flip(Xi_imgs[2], -1)
	                    Yi_imgs[1] = cv2.flip(Yi_imgs[1], -1)
	                    
	                X = np.expand_dims(np.concatenate((Xi_imgs[0],Xi_imgs[1],Xi_imgs[2]), axis=-1), axis=0)
	                y = np.expand_dims(Yi_imgs[1], axis=0)
	                
	                
	                X_out = np.concatenate((X_out, X))
	                y_out = np.concatenate((y_out, y))
	                
	                i += 1
	            
	        yield(X_out, y_out)                

	@threadsafe_generator          
	def val_generator(self):	    
	    while True:
	        i=0
	        X_out = np.empty((0, self.target_size[0], self.target_size[1], 9))
	        y_out = np.empty((0, self.target_size[0], self.target_size[1], 3))
	        while i  < self.batch_size: 
	            Xi = self.val_blur.next()
	            Xi_imgs = Xi[0]
	            Xi_labels = Xi[1]
	            Yi = self.val_sharp.next()
	            Yi_imgs = Yi[0]
	            Yi_labels = Yi[1]

	            if np.min(Xi_labels) == np.max(Xi_labels):
	                (Xi_imgs, Yi_imgs) = self.randomCrop(Xi_imgs, Yi_imgs, self.target_size[0], self.target_size[1], self.val_random)
	                
	                X = np.expand_dims(np.concatenate((Xi_imgs[0],Xi_imgs[1],Xi_imgs[2]), axis=-1), axis=0)
	                y = np.expand_dims(Yi_imgs[1], axis=0)
	                X_out = np.concatenate((X_out, X))
	                y_out = np.concatenate((y_out, y))
	                i+=1
	            
	        yield(X_out, y_out)
	
	@threadsafe_generator
	def test_generator(self):	    
	    while True:
	        i=0
	        X_out = np.empty((0, self.target_size[0], self.target_size[1], 9))
	        y_out = np.empty((0, self.target_size[0], self.target_size[1], 3))
	        while i  < self.batch_size: 
	            Xi = self.test_blur.next()
	            Xi_imgs = Xi[0]
	            Xi_labels = Xi[1]
	            Yi = self.test_sharp.next()
	            Yi_imgs = Yi[0]
	            Yi_labels = Yi[1]

	            if np.min(Xi_labels) == np.max(Xi_labels):
	                (Xi_imgs, Yi_imgs) = self.randomCrop(Xi_imgs, Yi_imgs, self.target_size[0], self.target_size[1], self.test_random)
	                
	                X = np.expand_dims(np.concatenate((Xi_imgs[0],Xi_imgs[1],Xi_imgs[2]), axis=-1), axis=0)
	                y = np.expand_dims(Yi_imgs[1], axis=0)
	                X_out = np.concatenate((X_out, X))
	                y_out = np.concatenate((y_out, y))
	                i+=1
	            
	        yield(X_out, y_out)