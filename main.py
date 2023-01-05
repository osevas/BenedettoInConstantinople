import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator


AUTOTUNE = tf.data.AUTOTUNE

def plot_sample_pictures():
	list_train_istanbul = os.listdir('./train_istanbul/')
	list_train_painting = os.listdir('./train_painting/')

	fig = plt.figure()

	for i in range(4):
		sp = plt.subplot(2, 4, i+1)
		sp.axis('Off')
		img = mpimg.imread('./train_istanbul/' + list_train_istanbul[i])
		plt.imshow(img)

		sp2 = plt.subplot(2, 4, i+5)
		sp2.axis('Off')
		img = mpimg.imread('./train_painting/' + list_train_painting[i])
		plt.imshow(img)
	plt.show()

# plot_sample_pictures()

# Instantiating ImageDataGenerator and rescaling
train_istanbul_datagen = ImageDataGenerator(rescale=1.0/255.)
test_istanbul_datagen = ImageDataGenerator(rescale=1.0/255.)

train_painting_datagen = ImageDataGenerator(rescale=1.0/255.)
test_painting_datagen = ImageDataGenerator(rescale=1.0/255.)

# Flow Istanbul images in batches of 20 using train_datagen generator
train_istanbul_generator = train_istanbul_datagen.flow_from_directory('./train_istanbul', batch_size=32, target_size=(256, 256), class_mode=None) # default is nearest for interpolation
test_istanbul_generator = test_istanbul_datagen.flow_from_directory('./test_istanbul', batch_size=32, target_size=(256, 256), class_mode=None)

# Flow Istanbul images in batches of 20 using train_datagen generator
train_painting_generator = train_painting_datagen.flow_from_directory('./train_painting', batch_size=32, target_size=(256, 256), class_mode=None)
test_painting_generator = test_painting_datagen.flow_from_directory('./test_painting', batch_size=32, target_size=(256, 256), class_mode=None)

# train_istanbul_ds = tf.data.Dataset.from_generator(train_istanbul_generator)
# print('shape of train_istanbul_ds: {}'.format(len(list(train_istanbul_ds.as_numpy_iterator()))))

# print(tf.shape(train_istanbul_generator.next()))

def plot_generator(gen1, gen2):
	fig = plt.figure()

	plt.subplot(121)
	plt.imshow(gen1[0][0])

	plt.subplot(122)
	plt.imshow(gen2[0][0])

	plt.show()

# plot_generator(train_istanbul_generator, train_painting_generator)

#------------------------------------------
# Import and reuse the Pix2Pix models
#------------------------------------------
OUTPUT_CHANNELS = 3

# x-> istanbul
# y-> painting

generator_i_p = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_p_i = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_i = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_p = pix2pix.discriminator(norm_type='instancenorm', target=False)

sample_istanbul = tf.expand_dims(train_istanbul_generator[0][0], 0)
sample_painting = tf.expand_dims(train_painting_generator[0][0], 0)

# print(sample_istanbul.shape)

to_painting = generator_i_p(sample_istanbul)
to_istanbul = generator_p_i(sample_painting)

def plot_x_to_y(images):
	plt.figure(figsize=(8, 8))
	contrast = 8

	imgs = images
	title = ['Istanbul', 'To Painting', 'Painting', 'To Istanbul']

	for i in range(len(imgs)):
		plt.subplot(2, 2, i+1)
		plt.title(title[i])
		if i % 2 == 0:
			plt.imshow(imgs[i][0] * 0.5 + 0.5)
		else:
			plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
	plt.show()

images = [sample_istanbul, to_painting, sample_painting, to_istanbul]
# plot_x_to_y(images)

def plot_is_real(pic_y, pic_x):
	plt.figure(figsize=(8, 8))

	plt.subplot(121)
	plt.title('Is a real zebra?')
	plt.imshow(discriminator_p(pic_y)[0, ..., -1], cmap='RdBu_r')

	plt.subplot(122)
	plt.title('Is a real horse?')
	plt.imshow(discriminator_i(pic_x)[0, ..., -1], cmap='RdBu_r')

	plt.show()

# plot_is_real(sample_painting, sample_istanbul)


#------------------------------------------
# Loss Functions
#------------------------------------------
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
	real_loss = loss_obj(tf.ones_like(real), real)

	generated_loss = loss_obj(tf.zeros_like(generated), generated)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss * 0.5

def generator_loss(generated):
  	return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
	loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
	
	return LAMBDA * loss1

def identity_loss(real_image, same_image):
	loss = tf.reduce_mean(tf.abs(real_image - same_image))
	return LAMBDA * 0.5 * loss

generator_i_p_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_p_i_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_i_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_p_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#------------------------------------------
# Checkpoints
#------------------------------------------
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_i_p,
                           generator_f=generator_p_i,
                           discriminator_x=discriminator_i,
                           discriminator_y=discriminator_p,
                           generator_g_optimizer=generator_i_p_optimizer,
                           generator_f_optimizer=generator_p_i_optimizer,
                           discriminator_x_optimizer=discriminator_i_optimizer,
                           discriminator_y_optimizer=discriminator_p_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
	ckpt.restore(ckpt_manager.latest_checkpoint)
	print ('Latest checkpoint restored!!')

#------------------------------------------
# Train
#------------------------------------------
EPOCHS = 40

def generate_images(model, test_input):
	prediction = model(test_input)
		
	plt.figure(figsize=(12, 12))

	display_list = [test_input[0], prediction[0]]
	title = ['Input Image', 'Predicted Image']

	for i in range(2):
		plt.subplot(1, 2, i+1)
		plt.title(title[i])
		# getting the pixel values between [0, 1] to plot it.
		plt.imshow(display_list[i] * 0.5 + 0.5)
		plt.axis('off')
	plt.show()

@tf.function
def train_step(real_i, real_p):
	# persistent is set to True because the tape is used more than
	# once to calculate the gradients.
	with tf.GradientTape(persistent=True) as tape:
		# Generator G translates X -> Y
		# Generator F translates Y -> X.
		
		fake_p = generator_i_p(real_i, training=True)
		cycled_i = generator_p_i(fake_p, training=True)

		fake_i = generator_p_i(real_p, training=True)
		cycled_p = generator_i_p(fake_i, training=True)

		# same_x and same_y are used for identity loss.
		same_i = generator_p_i(real_i, training=True)
		same_p = generator_i_p(real_p, training=True)

		disc_real_i = discriminator_i(real_i, training=True)
		disc_real_p = discriminator_p(real_p, training=True)

		disc_fake_i = discriminator_i(fake_i, training=True)
		disc_fake_p = discriminator_p(fake_p, training=True)

		# calculate the loss
		gen_i_p_loss = generator_loss(disc_fake_p)
		gen_p_i_loss = generator_loss(disc_fake_i)
		
		total_cycle_loss = calc_cycle_loss(real_i, cycled_i) + calc_cycle_loss(real_p, cycled_p)
		
		# Total generator loss = adversarial loss + cycle loss
		total_gen_i_loss = gen_i_p_loss + total_cycle_loss + identity_loss(real_p, same_p)
		total_gen_p_loss = gen_p_i_loss + total_cycle_loss + identity_loss(real_i, same_i)

		disc_i_loss = discriminator_loss(disc_real_i, disc_fake_i)
		disc_p_loss = discriminator_loss(disc_real_p, disc_fake_p)
	
	# Calculate the gradients for generator and discriminator
	generator_i_p_gradients = tape.gradient(total_gen_i_loss, 
											generator_i_p.trainable_variables)
	generator_p_i_gradients = tape.gradient(total_gen_p_loss, 
											generator_p_i.trainable_variables)
	
	discriminator_i_gradients = tape.gradient(disc_i_loss, 
												discriminator_i.trainable_variables)
	discriminator_p_gradients = tape.gradient(disc_p_loss, 
												discriminator_p.trainable_variables)
	
	# Apply the gradients to the optimizer
	generator_i_p_optimizer.apply_gradients(zip(generator_i_p_gradients, 
												generator_i_p.trainable_variables))

	generator_p_i_optimizer.apply_gradients(zip(generator_p_i_gradients, 
												generator_p_i.trainable_variables))
	
	discriminator_i_optimizer.apply_gradients(zip(discriminator_i_gradients,
													discriminator_i.trainable_variables))
	
	discriminator_p_optimizer.apply_gradients(zip(discriminator_p_gradients,
													discriminator_p.trainable_variables))


for epoch in range(EPOCHS):
	start = time.time()

	n = 0
	for image_x, image_y in zip(train_istanbul_generator, train_painting_generator):
		train_step(image_x, image_y)
		if n % 10 == 0:
			print ('.', end='')
		n += 1

	# clear_output(wait=True)
	# Using a consistent image (sample_horse) so that the progress of the model
	# is clearly visible.
	generate_images(generator_i_p, sample_istanbul)

	if (epoch + 1) % 5 == 0:
		ckpt_save_path = ckpt_manager.save()
		print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
															ckpt_save_path))

	print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
														time.time()-start))