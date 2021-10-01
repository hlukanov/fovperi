import numpy as np, math, sys, cv2, random, time;

# Reverse Map Pixel - fovY and fovX are the location which produced the maxX and maxY values
def reverseMapPixel( N, p0, Nr, bio=False ):
	bio_str = '';

	if bio:
		bio_str = '_bio';

	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);
	rev = np.load( './fov/' + conf_str + '-rev' + bio_str + '.npy' );

	shiftY = (N-p0*2)/2 + p0;
	shiftX = (N-p0*2)/2 + p0;

	def process( maxY, maxX, fovY, fovX ):
		return rev[maxY,maxX] + np.array( [fovY-shiftY,fovX-shiftX], dtype='uint8' );

	return process;

# Short function for when the initial location was used
def reverseMapPixel_Initial( N, p0, Nr, fovCenter, bio=False ):
	bio_str = '';

	if bio:
		bio_str = '_bio';

	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);
	rev = np.load( './fov/' + conf_str + '-rev' + bio_str + '.npy' );

	shiftY = int( (N-p0*2)/2 + p0 );
	shiftX = int( (N-p0*2)/2 + p0 );

	initialCoords = [int(fovCenter-shiftY),int(fovCenter-shiftX)];

	def process( maxY, maxX ):
		return rev[maxY,maxX] + initialCoords;

	return process;

def generate_fovea( N, p0, Nr ):
	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);

	mX = np.load( './fov/' + conf_str + '-mappingX.npy' );
	mY = np.load( './fov/' + conf_str + '-mappingY.npy' );

	dims = 2 * ( p0 + Nr ) + 1;

	jiggle_space_low = [];
	jiggle_space_high = [];

	center_of_fov = int(dims/2);
	mmX = mX.reshape( (dims,dims) );
	mmY = mY.reshape( (dims,dims) );

	line = mmY[center_of_fov,:];
	line_len = len( line );
	prev = 0;

	for i in range( line_len ):
		if i < center_of_fov:
			if line[i+1] == (line[i]+1):
				continue;

			dif = line[i]-prev;
			jiggle_space_low.append( dif );
			prev = line[i]+1;

		else:
			next = N;

			if i < line_len-1:
				next = line[i+1];

			if next == (line[i]+1) or next == line[i]:
				continue;

			dif = next - line[i] - 1;
			jiggle_space_high.append( dif );

	def augment_coordinates( arr ):
		for row in arr:
			prev = 0;
			next = 0;

			for i in range( len(row) ):
				if i < Nr:
					row[i] -= np.random.randint(jiggle_space_low[prev]+1,size=1);
					prev += 1;

				if i >= dims-Nr:
					row[i] += np.random.randint(jiggle_space_high[next]+1,size=1);
					next += 1;

	augment_coordinates( mmY );

	mmX = mmX.transpose();
	augment_coordinates( mmX );
	mmX = mmX.transpose();

	map = np.zeros( (dims,dims,1) );

	for i in range( dims ):
		for j in range( dims ):
			map[i,j,:] = np.sqrt( (N//2-mmX[i][j])**2 + (N//2-mmY[i][j])**2 );

	normalized = (map-np.min(map))/(np.max(map)-np.min(map));
	normalized = 1 - normalized; # invert, such that periphery is 0 and center is 1

	rev = np.zeros( (dims,dims,2) );
	rev[:,:,0] = mmX;
	rev[:,:,1] = mmY;

	mX = mmX.flatten();
	mY = mmY.flatten();

	np.save( './fov/' + conf_str + '-mappingX_bio.npy', mX );
	np.save( './fov/' + conf_str + '-mappingY_bio.npy', mY );
	np.save( './fov/' + str(dims) + 'x' + str(dims) + '_radial_normalized_bio.npy', (normalized*255).astype('int32')/255 );
	np.save( './fov/' + str(dims) + 'x' + str(dims) + '_radial_0-255_bio.npy', (normalized*255).astype('int32') );
	np.save( './fov/' + conf_str + '-rev_bio.npy', rev );

	cv2.imwrite( 'dist_map.jpg', (normalized*255).astype('int32') );

def add_dist_channel( batch_size, fovSize, bio=False ):
	bio_str = '_bio' if bio else '';

	template = np.load( './fov/81x81_radial_normalized' + bio_str + '.npy' );

	batch_template = np.zeros( (batch_size,fovSize,fovSize,1) );

	for i in range( batch_size ):
		batch_template[i] = template;

	def process( fovs, do_it ):
		if not do_it:
			return fovs;

		return np.concatenate( (fovs,batch_template), axis=3 );

	return process;

np.set_printoptions( threshold=sys.maxsize );

def getFov( N, p0, Nr, bio=False, add_coords=False ):
	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);

	bio_str = '';

	if bio:
		bio_str = '_bio';

	mappingX = np.load( './fov/' + conf_str + '-mappingX' + bio_str + '.npy' );
	mappingY = np.load( './fov/' + conf_str + '-mappingY' + bio_str + '.npy' );

	maxShift = np.load( './fov/' + conf_str + '-maxShift.npy' );
	minShift = -maxShift;

	dims = 2 * ( p0 + Nr ) + 1;
	maxVal = N-1;

	imDims = range( N );
	zdim = dims*dims;
	dim_l = ( np.array( list(range( zdim )) ), );

	abz = np.zeros( (zdim,), dtype='uint32' );

	cache = np.zeros( (213, 213, 3, dims * dims), dtype='uint32' );
	arr_len = np.zeros( (213,213), dtype='uint32' );
	ind_len = np.zeros( (213,213), dtype='uint32' );

	if add_coords:
		cache_coords = np.zeros( (213,213,dims,dims,1), dtype='int32' );

	map = np.zeros( (N,N,1) );

	for i in range( N ):
		for j in range( N ):
			map[i,j,:] = np.sqrt( (N//2-i)**2 + (N//2-j)**2 );

	map_normalized = (map-np.min(map))/(np.max(map)-np.min(map));
	map_normalized = 1 - map_normalized; # invert, so that center is 1 and periphery is 0 (otside is -1)
	map_0_255 = (map_normalized * 255).astype( 'int32' );

	for x in range(minShift,maxShift+1):
		for y in range(minShift,maxShift+1):
			mX = np.copy( mappingX ) + x;
			mY = np.copy( mappingY ) + y;

			ab = np.where( np.logical_and.reduce([mX >= 0,mY >= 0,mX <= maxVal,mY <= maxVal]) );

			mX = mX[ab];
			mY = mY[ab];

			ind_len[x][y] = len( ab[0] );
			arr_len[x][y] = len( mX );
			cache[x][y][0][:arr_len[x][y]] = mX;
			cache[x][y][1][:arr_len[x][y]] = mY;
			cache[x][y][2][:ind_len[x][y]] = ab[0];

			if add_coords:
				# Cache for coords
				fov = np.zeros( (zdim,1), dtype='int32' );
				fov -= 1;

				fov[ab,:] = map_0_255[mX,mY,:];
				fov=fov.reshape( (dims,dims,1) );
				cache_coords[x][y] = fov;

	def process( im, fovX=0, fovY=0, for_training=True, sample_matrix=False, add_coords=False, reverse=False ):
		fovX = min( max( fovX + minShift, minShift ), maxShift );
		fovY = min( max( fovY + minShift, minShift ), maxShift );

		if sample_matrix:
			orig_reconst = np.zeros_like( im, dtype='uint8' );

		channels = 3;

		fov = np.zeros( (zdim,channels), dtype='int32' );

		if for_training:
			fov -= 1;

		mX = cache[fovX][fovY][0][:arr_len[fovX][fovY]];
		mY = cache[fovX][fovY][1][:arr_len[fovX][fovY]];
		ab = cache[fovX][fovY][2][:ind_len[fovX][fovY]];

		if reverse:
			orig_reconst = np.zeros( (256,256,3), dtype='int32' );
			orig_reconst -= 10;
			print( im.shape );
			print( ab.shape );
			im = im.reshape( (zdim,channels) );
			orig_reconst[mX,mY,:] = im[ab,:];

			mask = (orig_reconst==-10)[:,:,:1].astype(np.uint8);

			orig_reconst[orig_reconst == -10] = 0;

			orig_reconst_interpolated = cv2.inpaint(np.uint8(orig_reconst), mask, 3, cv2.INPAINT_TELEA);

			return [orig_reconst,orig_reconst_interpolated];

		if sample_matrix:
			orig_reconst[mX,mY,:] = im[mX,mY,:];

		fov[ab,:] = im[mX,mY,:];
		fov=fov.reshape( (dims,dims,channels) );

		if add_coords:
			fov = np.dstack( (fov,cache_coords[fovX][fovY]) );

		if sample_matrix:
			return [fov,orig_reconst];
		else:
			return fov;

	return process;
