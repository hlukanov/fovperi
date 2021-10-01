import cv2, numpy as np, os, json, tensorflow as tf, random, shutil, math, time, h5py, SharedArray as sa, argparse;
from utils.preprocess import printProgressBar, printLoader;
from keras.backend.tensorflow_backend import set_session;

parser = argparse.ArgumentParser();
parser.add_argument( "--model", default='core50' );
parser.add_argument( "--task", default='train' );
args = parser.parse_args();

if args.model not in [ 'core50', 'imagenet' ]:
	print( 'ERROR: Model must be core50 or imagenet' );
	quit();

if args.task not in [ 'train', 'test' ]:
	print( 'ERROR: Task must be train or test' );
	quit();

print( '################################################################' );
print( 'Model: ' + str( args.model ) );
print( '\nTask: ' + str( args.task ) );
print( '################################################################' );
printLoader = printLoader( args.model );

my_index = 0;
other_index = 1;
load_data_flag = True;

if args.model == 'core50':
	add_coords = False;
	add_dist = True;
	bio_foveation = False;
else:
	add_coords = False;
	add_dist = False;
	bio_foveation = False;

tf.keras.backend.clear_session();
tf.config.optimizer.set_jit(True);
config = tf.ConfigProto(device_count={"CPU": 32});
config.gpu_options.allow_growth = False;
config.log_device_placement = False;
set_session(tf.Session(config=config));
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.FATAL );
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';

from tensorflow.keras import backend as K;
from xception import build_xception_imagenet, build_xception_core50;
from sklearn.utils import class_weight;
import numexpr as ne, gc;

from utils.training_utils import calculateStatistics, batch_generator, argmax_lastNaxes, step_decay;
from fov.fov2 import getFov, reverseMapPixel, reverseMapPixel_Initial, add_dist_channel;

# Batch size
test_batch_size = 1000;
batch_size = 32;
saccades = 1;
learning_rate = 0.001;

step_decay = step_decay( learning_rate );

# Foveation
N, p0, Nr = [256, 22, 18];
fovCenter = int( ( N - 2*p0 ) / 2 );

reverseMapPixel_Initial = reverseMapPixel_Initial( N=N, p0=p0, Nr=Nr, fovCenter=fovCenter, bio=bio_foveation );
reverseMapPixel = reverseMapPixel( N=N, p0=p0, Nr=Nr, bio=bio_foveation );
getFov = getFov( N=N, p0=p0, Nr=Nr, bio=bio_foveation, add_coords=add_coords );
fovSize = 2 * ( p0 + Nr ) + 1;

add_dist_channel_test = add_dist_channel( test_batch_size, fovSize=fovSize, bio=bio_foveation );
add_dist_channel = add_dist_channel( batch_size, fovSize=fovSize, bio=bio_foveation );

if len( sa.list() ) > 0:
	print( 'WARNING: SHARED MEMORY NOT EMPTY!\nClearing...\n' );
	mem_list = [ m.name.decode('utf-8') for m in sa.list() ];

	for mem_arr in mem_list:
		sa.delete( mem_arr );

models_config = sa.create( "shm://models", 3, dtype='int32' );
models_config[:] = np.zeros( 3, dtype='int32' );

models_config = sa.attach( 'shm://models' );
models_config[my_index] = 0;

if args.model == 'core50':
	model = build_xception_core50( batch_size, add_coords=add_coords, add_dist=add_dist, additional_blocks=1, learning_rate=learning_rate );
	test_model = build_xception_core50( test_batch_size, add_coords=add_coords, add_dist=add_dist, additional_blocks=1, learning_rate=learning_rate );
else:
	model = build_xception_imagenet(batch_size);
	test_model = build_xception_imagenet(test_batch_size);

tf.keras.utils.plot_model( model, to_file='model.png', show_shapes=True, show_layer_names=True );

Statter = calculateStatistics( model, log_time = True, gazes=saccades );
TestStatter = calculateStatistics( model, log_time = True, gazes=saccades );

dataset = h5py.File( "./dataset/" + args.model + ".hdf5", "r" );

if args.model == 'core50':
	labels = 'labels50';
else:
	labels = 'labels';

y_train = dataset['train/' + labels][:].flatten().tolist();
class_weights = class_weight.compute_class_weight( 'balanced', np.unique( y_train ), y_train );
del y_train;

###############################################################################################################################################
gen_batch = batch_generator(
	dataset['train/images'],
	dataset['train/' + labels],
	batch_size=batch_size,
	buffer_size=80000,
	aug = True,
	cpus=32,
	shm_name = 'train_data',
	add_coords = add_coords,
	load_data=load_data_flag,
	my_index = my_index,
	other_index = other_index,
	bio=bio_foveation,
	multi_model=False
);

test_gen_batch = batch_generator(
	dataset['test/images'],
	dataset['test/' + labels],
	batch_size=test_batch_size,
	buffer_size=50000,
	aug = None,
	cpus=32,
	shm_name = 'test_data',
	add_coords = add_coords,
	load_data=load_data_flag,
	my_index = my_index,
	other_index = other_index,
	bio=bio_foveation,
	multi_model=False
);

s_time = time.time();

iters = 0;
normalizer = np.float32( 255 );

epochs = 50;

fov2 = np.zeros( (batch_size,fovSize,fovSize,4 if add_coords else 3), dtype='float32' );
fov2_test = np.zeros( (test_batch_size,fovSize,fovSize,4 if add_coords else 3), dtype='float32' );

# Train
if args.task == 'train':
	for e in range( 0, epochs ):
		K.set_value( model.optimizer.learning_rate, step_decay( e ) );

		s_time = time.time();

		gen_batch.restart();
		Statter.restart();
		iters = 0;

		print( 'Epoch ' + str(e+1) + '/' + str(epochs) + ' | Model: ' + args.model + '\n--------------------------' );

		history = {
			'loss': {},
			'acc': {}
		};

		for i in range( saccades+1 ):
			history['loss']['g'+str(i)] = [];
			history['acc']['g'+str(i)] = [];

		bseen = 0;

		while( True ):
			X, Y, fov = gen_batch.next( normalize=True, use_stats=False );

			if isinstance( X, bool ):
				break;

			if add_dist:
				fov = add_dist_channel( fov, add_dist );

			training_stats = model.train_on_batch( fov, Y );
			bufs = training_stats[-1];
			training_stats = training_stats[:-1];

			history['loss']['g0'].append( float(training_stats[0]) );
			history['acc']['g0'].append( float(training_stats[1]) );

			Statter.update_stats( training_stats, 0 );

			for saccade in range( saccades ):
				for i in range( batch_size ):
					locs = reverseMapPixel_Initial( bufs[0][i], bufs[1][i] );
					fov2[i] = getFov( X[i], fovY=int(locs[1]), fovX=int(locs[0]), add_coords=add_coords ); # fovY, fovX are switched, because argmax_lastNaxes returns y,x

				fov2 = ne.evaluate( 'fov2/normalizer' );

				fov_input = add_dist_channel( fov2, add_dist );

				training_stats = model.train_on_batch( fov_input, Y );
				training_stats = training_stats[:-1];

				history['loss']['g'+str(saccade+1)].append( float(training_stats[0]) );
				history['acc']['g'+str(saccade+1)].append( float(training_stats[1]) );

				Statter.update_stats( training_stats, saccade+1 );

			iters += 1;

			if iters % 50 == 0:
				cur, total = gen_batch.current_progress();
				loss_string = Statter.getLossString( cur, total );

				printProgressBar( cur, total, prefix = str(cur) + '/' + str( total ), suffix = '| ' + loss_string );

		print( '\n' );

		del X,Y,fov;
		gen_batch.free_memory();

		model.save_weights( './weights/' + args.model + '_e' + str(e+1) + '.h5' );

		test_model.set_weights( model.get_weights() );

		e_time = time.time();
		print( '\n\n\nTIME:' );
		print( e_time - s_time );

# Test
if args.task == 'test':
	test_model.load_weights( './weights/' + args.model + '.h5' );

	test_gen_batch.restart();
	TestStatter.restart();

	iters = 0;

	while( True ):
		X, Y, fov = test_gen_batch.next( normalize=True, use_stats=False );

		if isinstance( X, bool ):
			break;

		fov_input = add_dist_channel_test( fov, add_dist );

		test_stats = test_model.evaluate( fov_input, Y, verbose=0 );
		bufs = test_stats[-1];
		test_stats = test_stats[:-1];

		TestStatter.update_stats( test_stats, 0 );

		for saccade in range( saccades ):
			for i in range( test_batch_size ):
				locs = reverseMapPixel_Initial( bufs[0][i], bufs[1][i] );
				fov2_test[i], reconst = getFov( X[i], fovY=int(locs[1]), fovX=int(locs[0]), add_coords=add_coords, sample_matrix=True ); # fovY, fovX are switched, because argmax_lastNaxes returns y,x

			fov2_test = ne.evaluate( 'fov2_test/normalizer' );

			fov_input = add_dist_channel_test( fov2_test, add_dist );

			test_stats = test_model.evaluate( fov_input, Y, verbose=0 );
			bufs = test_stats[-1];
			test_stats = test_stats[:-1];

			TestStatter.update_stats( test_stats, saccade+1 );

		iters += 1;

		if iters % 100 == 0:
			cur, total = test_gen_batch.current_progress();
			loss_string = TestStatter.getLossString( cur, total );

			printProgressBar( cur, total, prefix = str(cur) + '/' + str( total ), suffix = '| ' + loss_string );

	loss_string = TestStatter.getLossString( total, total );
	print( loss_string + '                                                                     \n' );
























quit();
