######################################################
# %% Import modules

import json
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Dropout, Convolution2D
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


# %% Convenience functions

######################################################
# Serialize CNN model architecture and weights

def serialize_model_weights(model, fname="model"):
	"""
	Serializes Tensorflow/Keras model architecture and save weights
	:param model: Tensorflow/Keras model object
	:param fname: File name
	:return:
	    None
	"""

	# serialize model to JSON
	model_json = model.to_json()
	with open(fname + ".json", 'w') as f:
		json.dump(model_json, f)

	# serialize weights to HDF5
	model.save_weights(fname + ".h5")
	print("Saved model to disk")


# %%
######################################################
# Define CNN architecture
def commaai_architecture(img_shape, dropout=0.25):
	"""
	Defines CNN architecture
	:param img_shape: image shape
	:param dropout: value of dropout
	:return:
	    Tensorflow/Keras model summary and object
	"""

	# Adopted from Comma.ai: https://github.com/commaai/research/blob/master/train_steering_model.py

	padding = 'same'
	activationf = 'elu'

	model = Sequential()

	# Lambda - Normalization Layer
	model.add(Lambda(lambda x: x / 127.5 - 1.0, name='lambda_01', input_shape=img_shape))

	# Convolution Layer 1
	model.add(Convolution2D(32, 3, 3, border_mode=padding, activation=activationf, subsample=(2, 2), name='conv1_1'))

	# Convolution Layer 2
	model.add(Convolution2D(64, 3, 3, border_mode=padding, name='conv2_1', activation=activationf, subsample=(2, 2)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	# Convolution Layer 3
	model.add(Convolution2D(128, 3, 3, border_mode=padding, name='conv3_1', activation=activationf, subsample=(2, 2)))

	# Flatten
	model.add(Flatten())
	model.add((Dropout(dropout)))
	model.add(Activation(activationf))

	# Dense Layer
	model.add(Dense(512, activation=activationf))
	model.add((Dropout(dropout)))

	# Dense Layer
	model.add(Dense(1))

	# Model Summary
	model.summary()

	return model


######################################################
# %%  Split Data - Train/Valid
def train_valid_test_split(df, random_state=0, size_train=0.8, size_valid=0.2):
	"""
	Split Data - Train/Valid dataset
	:param df: data frame
	:param random_state: random state seed value
	:param size_train: fraction of training sample
	:param size_valid: fraction of validation sample
	:return:
	    training and validation dataset
	"""

	df_train, df_valid = train_test_split(df, test_size=size_valid, random_state=random_state)

	return df_train, df_valid


######################################################
# %%  read batch data
def read_batchdata(df,
                   bright=0,
                   brightness_perc=0.25,
                   crop=1,
                   resize=1,
                   resize_height=40,
                   resize_width=80,
                   normalize=0,
                   flip=0):
	"""
	Split Data - Train/Valid dataset
	:param df: data frame
	:param bright: binary value to augment brigtness in an image
	:param brightness_perc: fraction of brightness_percentage
	:param crop: binary value to crop an image
	:param resize: binary value to resize an image
	:param resize_height: # resized pixels in height
	:param resize_width: # resized pixels in width
	:param normalize: binary value to augment normalize an image
	:param flip: binary value to flip an image
	:return:
	    batch of images and steering angles
	"""

	# read images
	imgs = df['image'].values
	steering = df['steering'].values
	x_translation = df['x_translation'].values

	cv_img = []
	steeringn = []

	img = cv2.imread(imgs[0].strip())
	(rows, cols, channels) = img.shape

	# Read camera images in numpy array
	i = 0
	for img in imgs:

		# read image
		img = cv2.imread(img.strip())

		# translate image in X and Y direction
		tr_x = x_translation[i]
		tr_y = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)
		trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
		img = cv2.warpAffine(img, trans_m, (cols, rows))

		# augment brightness
		if bright == 1:
			imgtemp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			random_bright = brightness_perc + np.random.uniform()
			imgtemp[:, :, 2] = imgtemp[:, :, 2] * random_bright
			img = cv2.cvtColor(imgtemp, cv2.COLOR_HSV2RGB)

		# crop Images
		if crop == 1:
			img = img[60:140, :, :]

		# Resize images
		if resize == 1:
			img = cv2.resize(img, (resize_width, resize_height))  # , interpolation=cv2.INTER_AREA)

		# normalize images
		if normalize == 1:
			img = (img / 127.5 - 1.0)

		# flip image
		if flip == 1:
			if np.random.randint(2) == 0:  # Flip the image
				img = np.asarray(cv2.flip(img, 1))
				steeringn.append(-steering[i])
			else:
				steeringn.append(steering[i])
		else:
			steeringn.append(steering[i])

		cv_img.append(img)
		i += 1

	imgs = np.asarray(cv_img).astype('uint8')

	return imgs, np.asarray(steeringn)


########################################################
# %% valid/test data tuples
def valid_test_data(df):
	"""
	Read valid/test
	:param df: data frame
	:return:
	    images and steering angles
	"""
	features, labels = read_batchdata(df,
	                                  bright,
	                                  brightness_perc,
	                                  crop,
	                                  resize,
	                                  resize_height,
	                                  resize_width,
	                                  normalize,
	                                  flip=1)

	return features, labels


#######################################################
# %% plot loss history
def plot_loss(history_loss, history_vloss, fname='loss.png'):
	"""
	Plot loss values over epochs
	:param history_loss: list of loss values for training set
	:param history_vloss: list of loss values for validation set
	:param fname: File name
	:return:
	    Saves training/valid loss over epochs
	"""

	fig, ax = plt.subplots()
	plt.grid(True)
	plt.plot(history_loss, '-r', linewidth=2)
	plt.plot(history_vloss, '-b', linewidth=2)
	plt.title('model loss')
	plt.ylabel('loss (mse)', fontsize=14)
	plt.xlabel('epoch', fontsize=14)
	plt.legend(['train', 'valid'], loc='upper right')
	fig.savefig(fname + '_loss.png', dpi=100, bbox_inches='tight')


# %% Save prediction and true values in a csv file
def save_prediction(features, labels, fname):
	"""
	Save prediction and actual steering angles in a CSV file
	:param features: features
	:param labels: actual steering angles
	:param fname: File name
	:return:training/valid loss over epochs
	    Saves prediction and actual steering angles in a CSV file
	"""

	y_pred = model.predict(features)
	labels_true = np.reshape(labels, (len(labels), 1))
	hstack_labels = np.hstack((labels_true, y_pred))
	np.savetxt(fname + '_ypred_true.csv', hstack_labels, delimiter=',')

	x = range(len(y_pred))
	fig, ax = plt.subplots()
	plt.grid(True)
	plt.plot(x, labels_true - y_pred, '-r', linewidth=1)
	plt.axis([0, 500, -1, 1])
	plt.ylabel('error', fontsize=14)
	plt.xlabel('samples', fontsize=14)
	fig.savefig(fname + '_error_500.png', dpi=100, bbox_inches='tight')

	fig, ax = plt.subplots()
	plt.grid(True)
	plt.plot(x, labels_true - y_pred, '-r', linewidth=1)
	# plt.axis([0, 500, -1, 1])
	plt.ylabel('error', fontsize=14)
	plt.xlabel('samples', fontsize=14)
	fig.savefig(fname + '_error_full.png', dpi=100, bbox_inches='tight')


#########################################################
# %% Read generated log file

def read_csv_df(directory, fname_log_file):
	"""
	Read CSV file in Pandas Dataframe
	:param directory: Directory where log file is saved
	:param fname_log_file: log file name
	:return:
	    pandas dataframe with stacked
	"""
	df = pd.read_csv(directory + fname_log_file)  # , names=columns)

	# Stack center, left and right images in dataframe
	dfc = df[['center', 'steering']]
	dfc['camera'] = 'c'
	dfc.rename(columns={'center': 'image'}, inplace=True)

	dfl = df[['left', 'steering']]
	dfl['steering'] += offset
	dfl['camera'] = 'l'
	dfl.rename(columns={'left': 'image'}, inplace=True)

	dfr = df[['right', 'steering']]
	dfr['steering'] = dfr['steering'] - offset
	dfr['camera'] = 'r'
	dfr.rename(columns={'right': 'image'}, inplace=True)

	frames = [dfc, dfl, dfr]
	dfn = pd.concat(frames)

	dfn = dfn.loc[np.abs(dfn['steering']) <= 1.0]
	dfn = dfn.reset_index(drop=True)
	return dfn


#########################################################
# %% Sample dataframe
def sample_df(df, samples):
	"""
	Read CSV file in Pandas Dataframe
	:param df: data frame
	:param samples: batch sample size in integer
	:return:
	    sampled pandas dataframe with stacked
	"""

	dfc = df.loc[df['camera'] == 'c']
	dfl = df.loc[df['camera'] == 'l']
	dfr = df.loc[df['camera'] == 'r']

	dfcsamples = dfc.sample(int(samples / 3))
	dflsamples = dfl.sample(int(samples / 3))
	dfrsamples = dfr.sample(samples - 2 * int(samples / 3))

	frames = [dfcsamples, dflsamples, dfrsamples]
	dfn = pd.concat(frames)
	return dfn


########################################################
# %% # biased batch data

def biased_batch_data(df, batch_size, bias):
	"""
	Read CSV file in Pandas Dataframe
	:param df: data frame
	:param samples: batch sample size in integer
	:return:
	    sampled pandas dataframe with stacked
	"""

	# pick data equally from c, l and r images
	nbatches = 4

	dfsamples = shuffle(sample_df(df, samples=nbatches * batch_size))
	dftemp = pd.DataFrame(columns=['image', 'steering', 'camera', 'x_translation'])

	for index, row in dfsamples.iterrows():
		x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
		new_angle = row['steering'] + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE

		row['steering'] = new_angle
		row['x_translation'] = x_translation

		threshold = np.random.uniform()
		if ((abs(new_angle) + bias) > threshold):
			dftemp.loc[len(dftemp.index)] = row
			if len(dftemp.index) > batch_size + 50:
				break

	dftemp = dftemp.loc[np.abs(dftemp['steering']) <= 1.0]
	dftemp = dftemp.sample(batch_size)

	return dftemp


########################################################
# %% # Data Generator

def data_generator(df, batch_size, bias):
	"""
	Read batch data using the generator
	:param df: data frame
	:param batch_size: batch size in integer
	:param bias: bias value to select right/left turn steering angles
	:return:
	    batch images and steering angles
	"""

	total_batch = int(len(df.index) / batch_size)
	while True:
		for i in range(total_batch):
			# sample bias batch data
			datadf = biased_batch_data(df, batch_size, bias)

			# read batch images and labels
			batch_images, batch_labels = read_batchdata(datadf,
			                                            bright,
			                                            brightness_perc,
			                                            crop,
			                                            resize,
			                                            resize_height,
			                                            resize_width,
			                                            normalize,
			                                            flip=1)

			yield (batch_images, batch_labels)


#########################################################
if __name__ == '__main__':

	# Set the seed for predictability
	randomseed = 200
	np.random.seed(randomseed)

	directory = '/home/rakesh/SDC/SDC-ND/07_BehavioralCloning/Simulator10Hz/Track1_Udacity/'
	fname_log_file = 'driving_log.csv'
	offset = 0.25

	# define global variables
	global bright, brightness_perc, crop, resize, resize_height, resize_width, normalize

	bright = 1
	brightness_perc = 0.25
	crop = 1
	resize = 1
	resize_height = 40  # 66
	resize_width = 80  # 200
	normalize = 0

	TRANS_X_RANGE = 100  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
	TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
	TRANS_ANGLE = 0.3

	#########################################################
	# added random state
	df = read_csv_df(directory, fname_log_file)
	df_train, df_valid, df_test = train_valid_test_split(df,
	                                                     random_state=randomseed,
	                                                     size_train=0.8,
	                                                     size_valid=0.2)

	#########################################################
	# Get sample image
	batch_size = 1
	dftemp = df_train
	data = dftemp.sample(batch_size, replace=False, random_state=0)
	data['x_translation'] = 0

	aimgs, _ = read_batchdata(data,
	                          bright,
	                          brightness_perc,
	                          crop,
	                          resize,
	                          resize_height,
	                          resize_width,
	                          normalize,
	                          flip=1)

	#########################################################

	# Define flags to pass values as command line arguments to the program

	flags = tf.app.flags
	FLAGS = flags.FLAGS

	# command line flags
	flags.DEFINE_integer('nb_epoch', 10, "The number of epochs.")
	flags.DEFINE_integer('batch_size', 256, "The batch size.")
	flags.DEFINE_float('learning_rate', 0.001, "The learning rate.")
	flags.DEFINE_float('dropout', 0.25, "Dropout")
	flags.DEFINE_string('fname_model', 'model', "File name of the model")

	# Define CNN model

	img_shape = aimgs[0].shape
	model = commaai_architecture(img_shape, FLAGS.dropout)

	# Compile Model
	optimizer = Adam(lr=FLAGS.learning_rate)
	model.compile(loss='mse', optimizer=optimizer)

	#################################################

	num_runs = 0

	# valid, test dataset
	df_valid['x_translation'] = 0

	# read validation dataset
	features_valid, labels_valid = valid_test_data(df_valid)

	# Define empty list to store loss values after each epoch
	history_loss = []  # training loss
	history_vloss = []  # validation loss

	while True:
		bias = 1.0 / (num_runs + 1.0)

		print('\nRun {} with bias {}'.format(num_runs + 1, bias), end=': ')

		history = model.fit_generator(
			data_generator(df_train, FLAGS.batch_size, bias),
			samples_per_epoch=len(df_train.index),
			nb_epoch=1,
			validation_data=data_generator(df_valid, FLAGS.batch_size, bias=1),
			nb_val_samples=len(df_valid.index),
			verbose=1
		)

		# Save the model and the weights
		serialize_model_weights(model, FLAGS.fname_model + str(num_runs))

		# append history loss
		history_loss.append(history.history['loss'][0])
		history_vloss.append(history.history['val_loss'][0])

		num_runs += 1

		# Terminate epoch
		if num_runs > FLAGS.nb_epoch:
			break

	# Plot history for loss for training and validation data set
	plot_loss(history_loss, history_vloss, fname=FLAGS.fname_model)

	# Save prediction values for valid dataset in a CSV file
	save_prediction(features_valid, labels_valid, fname=FLAGS.fname_model + '_valid')

	# Print score for validation dataset
	score_valid = model.evaluate(features_valid, labels_valid)
	print('Test score:', score_valid)

	print(">>>>>>>> Done <<<<<<<<<<")
