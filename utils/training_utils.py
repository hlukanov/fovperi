import time, pickle, json, SharedArray as sa, h5py, math, numpy as np, numexpr as ne, gc, sys, cv2;

from fov.fov2 import getFov, reverseMapPixel, reverseMapPixel_Initial;
from utils.preprocess import printProgressBar, printLoader;
from multiprocessing import Process;
from albumentations import Compose;
import albumentations as A;


# Foveation
N, p0, Nr = [256, 22, 18];
fovCenter = int( ( N - 2*p0 ) / 2 );

getFov_bio = getFov( N=N, p0=p0, Nr=Nr, bio=True );
getFov = getFov( N=N, p0=p0, Nr=Nr, bio=False );
fovSize = 2 * ( p0 + Nr ) + 1;

aug = Compose( [
	A.ShiftScaleRotate( shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.7 ),
	A.HorizontalFlip( p=0.5 )
], p=1.0 );

def argmax_lastNaxes(A, N):
    s = A.shape;
    new_shp = s[:-N] + (np.prod(s[-N:]),);
    max_idx = A.reshape(new_shp).argmax(-1);
    return np.unravel_index(max_idx, s[-N:]);

def step_decay( initial_lrate = 0.045 ):

	def give_step( epoch ):
		drop = 0.94;
		epochs_drop = 2.;
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop));

		return lrate;

	return give_step;

def augment( start, to, shared_name ):
	b = sa.attach( 'shm://' + shared_name );

	for im_n in range( start, to ):
		b[im_n] = aug(image=b[im_n])['image'];

	del b;
	b = None;

def foveateInitialLocations( start, to, shared_name, bio=False ):
	add = 'bio_' if bio else '';
	a = sa.attach( 'shm://' + shared_name );
	b = sa.attach( 'shm://fov_' + add + shared_name );

	fn = getFov_bio if bio else getFov;

	for im_n in range( start, to ):
		b[im_n] = fn( a[im_n], fovY=fovCenter, fovX=fovCenter );

	del a, b;
	a = None;
	b = None;

def get_steps( batch_size, cpus=16 ):
	amounts = [];
	step = int(math.ceil(batch_size/cpus));

	if batch_size % cpus == 0:
		amounts = [step]*int(cpus);
	else:
		amounts = [step]*int(cpus);
		amounts[-1] = int(step - (step*cpus-batch_size));

	return [step,amounts];

class calculateStatistics:
	def __init__( self, model, log_time = False, gazes = 0 ):
		self.meta_dict = [];
		self.log_time = log_time;
		self.stats = {};
		self.gazes = gazes + 1;

		def replaces( name ):
			name = name.replace( 'acc', 'a' ).replace( 'loss', 'l' );
			return name;

		for i in range( self.gazes ):
			self.meta_dict.append( [] );

			for k in model.metrics_names:
				if k == 'att_map':
					continue;
				self.meta_dict[i].append( replaces(k) + '_g' + str(i+1) );

		self.s_time = time.time();
		self.time_steps = 0.;
		self.time_sum = 0.;

		for i in range( self.gazes ):
			for name in self.meta_dict[i]:
				self.stats[name] = { 'total': 0, 'steps': 0 };

	def restart( self ):
		self.s_time = time.time();
		self.stats = {};
		self.time_steps = 0.;
		self.time_sum = 0.;

		for i in range( self.gazes ):
			for name in self.meta_dict[i]:
				self.stats[name] = { 'total': 0, 'steps': 0 };

	def update_stats( self, returns, gaze ):
		for i in range(len(self.meta_dict[gaze])):
			self.stats[self.meta_dict[gaze][i]]['total'] += returns[i];

		if self.log_time and gaze == 0:
			self.time_sum += time.time() - self.s_time;
			self.time_steps += 1.;
			self.s_time = time.time();

	def getLossString( self, cur, total ):
		loss_string = [];

		for g in range( self.gazes ):
			for i in range(len(self.meta_dict[g])):
				loss_string.append( self.meta_dict[g][i] + ': %.4f' % round( self.stats[self.meta_dict[g][i]]['total']/cur, 4 ) );

		if self.log_time:
			loss_string.append( 'epoch ETA: ' + str( round( ( self.time_sum / self.time_steps ) * total ) ) + ' sec                             ' );

		return ', '.join( loss_string );


class batch_generator:
	def __init__( self, X, Y, batch_size, buffer_size, statistics = [None,None], aug = None, shuffle = True, shm_name = None, cpus = 32, class_weights = None, add_coords = False, load_data = True, my_index = None, other_index = None, multi_model=True, bio=False ):
		X_len = len( X );
		Y_len = len( Y );

		assert X_len == Y_len;

		assert isinstance( shm_name, str );
		self.shm_name = shm_name + '_' + str(buffer_size);

		self.total_batches = math.floor( Y_len/batch_size );
		self.models_config = sa.attach( "shm://models" );
		self.class_weights = class_weights;
		self.mean, self.std = statistics;
		self.buffer_size = buffer_size;
		self.batch_size = batch_size;
		self.add_coords = add_coords;
		self.other_index = other_index;
		self.my_index = my_index;
		self.load_data = load_data;
		self.shuffle = shuffle;
		self.finished = False;
		self.current_buffer = 0;
		self.current_batch = 0;
		self.seed_counter = 0;
		self.current_total_batch = 0;
		self.multi_model = multi_model;
		self.printLoader = printLoader( 'A' if my_index==0 else 'B' );
		self.cpus = float(cpus);
		self.aug = aug;
		self.bio = bio;

		if self.add_coords:
			bio_str = '_bio' if bio else '';

			self.initial_coords_template = np.load( './fov/81x81_radial_normalized' + bio_str + '.npy' );
			self.batch_coords = np.array( [self.initial_coords_template,]*batch_size );

		self.X = X;
		self.Y = Y;

		self.buf_data = {
			'X': None,
			'Y': None,
			'fov': None
		};

		self.buffer_list = list(range(math.ceil( X_len/buffer_size )));


	def free_memory( self ):
		self.buf_data = { 'X': None, 'Y': None, 'fov': None };

		if self.load_data:
			mem_list = [ m.name.decode('utf-8') for m in sa.list() ];

			if self.shm_name in mem_list:
				sa.delete( self.shm_name );
				sa.delete( 'fov_' + self.shm_name );
				sa.delete( 'fov_bio_' + self.shm_name );
				sa.delete( 'ys_' + self.shm_name );

		self.sync( 'Freeing memory...' );

	def current_progress( self ):
		return [self.current_total_batch, self.total_batches];

	def finish( self ):
		printProgressBar( self.total_batches, self.total_batches, prefix = str(self.total_batches) + '/' + str( self.total_batches ) );
		self.finished = True;

	def get_steps( self ):
		buf_len = len(self.buf_data['Y']);

		amounts = [];
		step = int(math.ceil(buf_len/self.cpus));

		if buf_len % self.cpus == 0:
			amounts = [step]*int(self.cpus);
		else:
			amounts = [step]*int(self.cpus);
			amounts[-1] = int(step - (step*self.cpus-buf_len));

		return [buf_len,step,amounts];

	def augment_all( self, buf_len, step, amounts ):
		procs = [];

		c = 0;
		for i in range( 0, buf_len, step ):
			proc = Process( target=augment, args=(i,i+amounts[c],self.shm_name) );
			procs.append( proc );
			proc.start();
			c += 1;

		for proc in procs:
			proc.join();

	def initial_foveate_all( self, buf_len, step, amounts, bio=False ):
		procs = [];

		c = 0;
		for i in range( 0, buf_len, step ):
			proc = Process( target=foveateInitialLocations, args=(i,i+amounts[c],self.shm_name, bio) );
			procs.append( proc );
			proc.start();
			c += 1;

		for proc in procs:
			proc.join();

	def next( self, normalize=True, use_stats=False ):
		if self.finished:
			return [False,False,False];

		b = self.current_batch;

		b_from = b*self.batch_size;
		b_to = b_from + self.batch_size;

		fov = self.buf_data['fov'][b_from:b_to];
		X = self.buf_data['X'][b_from:b_to];
		Y = self.buf_data['Y'][b_from:b_to];

		if normalize:
			normalizer = np.float32( 255 );
			fov = ne.evaluate( 'fov/normalizer' );

			if self.add_coords:
				fov = np.concatenate( (fov,self.batch_coords), axis=3 );

		self.current_total_batch += 1;
		self.current_batch += 1;

		if self.current_batch >= self.current_buffer_total_batches:
			if self.current_buffer == len(self.buffer_list)-1:
				self.finish();
			else:
				self.current_buffer += 1;
				self.current_batch = 0;

				self.sync( 'Loading next buffer...' );

				if self.load_data:
					self.load_buffer();

				self.sync( 'Next buffer loaded...' );
				self.current_buffer_total_batches = self.models_config[2];

				if self.load_data:
					buf_len, step, amounts = self.get_steps();

					if self.aug is not None:
						self.augment_all( buf_len, step, amounts );

					self.initial_foveate_all( buf_len, step, amounts, bio=False );
					self.initial_foveate_all( buf_len, step, amounts, bio=True );

				self.sync( 'Augment and foveate next buffer...' );

		if self.class_weights is not None:
			Y_len = len(Y);
			class_weights = np.zeros( (Y_len,), dtype='float32' );

			for i in range(Y_len):
				class_weights[i] = self.class_weights[Y[i][0]];

		return [ X, Y, fov ];

	def load_buffer( self, step=50000 ):
		start_ind = self.buffer_list[self.current_buffer]*self.buffer_size;
		end_ind = start_ind + self.buffer_size;

		if self.Y.shape[0] < end_ind:
			end_ind = self.Y.shape[0];
		ys_len = end_ind - start_ind;

		if ys_len % self.batch_size > 0:
			end_ind -= ys_len % self.batch_size;
			ys_len = end_ind - start_ind;

		for i in range( 0, ys_len, step ):
			end_ = i+step;

			if end_ > ys_len:
				end_ = ys_len;

			self.buf_data['Y'][i:end_] = self.Y[start_ind+i:start_ind+end_];
			self.buf_data['X'][i:end_] = self.X[start_ind+i:start_ind+end_];

		# CURRENT BUFFER TOTAL BATCHES
		self.models_config[2] = math.floor( ys_len/self.batch_size );

	def sync( self, message, sync_sleep=1.5, check_every=0.5 ):
		self.printLoader( message );

		if self.multi_model:
			while not np.all( self.models_config[:2] ):
				self.printLoader( message );
				self.models_config[self.my_index] = 1;
				time.sleep( check_every );

			time.sleep( sync_sleep );
			self.models_config[self.my_index] = 0;

	def restart( self ):
		self.current_total_batch = 0;
		self.current_buffer = 0;
		self.current_batch = 0;
		self.finished = False;

		if self.load_data:
			np.random.shuffle( self.buffer_list );

		self.sync( 'Buffer creation...' );

		if self.load_data:
			mem_list = [ m.name.decode('utf-8') for m in sa.list() ];

			if self.shm_name not in mem_list:
				self.buf_data['X'] = sa.create( "shm://" + self.shm_name, (self.buffer_size,) + self.X.shape[1:], dtype='uint8' );
				self.buf_data['Y'] = sa.create( "shm://ys_" + self.shm_name, (self.buffer_size,50), dtype='uint8' );
				sa.create( "shm://fov_" + self.shm_name, (self.buffer_size,fovSize,fovSize,3), dtype='int32' );
				sa.create( "shm://fov_bio_" + self.shm_name, (self.buffer_size,fovSize,fovSize,3), dtype='int32' );

			self.load_buffer();

		self.sync( 'Buffer loaded...' );
		self.current_buffer_total_batches = self.models_config[2];

		bio_str = 'bio_' if self.bio else '';

		self.buf_data['X'] = sa.attach( "shm://" + self.shm_name );
		self.buf_data['Y'] = sa.attach( "shm://ys_" + self.shm_name );
		self.buf_data['fov'] = sa.attach( "shm://fov_" + bio_str + self.shm_name );

		if self.load_data:
			buf_len, step, amounts = self.get_steps();

			if self.aug is not None:
				self.augment_all( buf_len, step, amounts );

			self.initial_foveate_all( buf_len, step, amounts, bio=False );
			self.initial_foveate_all( buf_len, step, amounts, bio=True );

		self.sync( 'Augment and foveate...' );
