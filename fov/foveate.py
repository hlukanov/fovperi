import numpy as np, math, sys, cv2, random, time;

# Reverse Map Pixel - fovY and fovX are the location which produced the maxX and maxY values
def reverseMapPixel( N, p0, Nr ):
	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);
	rev = np.load( './fov/' + conf_str + '-rev.npy' );

	shiftY = (N-p0*2)/2 + p0;
	shiftX = (N-p0*2)/2 + p0;

	def process( maxY, maxX, fovY, fovX ):
		return rev[maxY,maxX] + np.array( [fovY-shiftY,fovX-shiftX], dtype='uint8' );

	return process;

# Short function for when the initial location was used
def reverseMapPixel_Initial( N, p0, Nr, fovCenter ):
	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);
	rev = np.load( './fov/' + conf_str + '-rev.npy' );

	shiftY = int( (N-p0*2)/2 + p0 );
	shiftX = int( (N-p0*2)/2 + p0 );

	initialCoords = [int(fovCenter-shiftY),int(fovCenter-shiftX)];

	def process( maxY, maxX ):
		return rev[maxY,maxX] + initialCoords;

	return process;

np.set_printoptions( threshold=sys.maxsize );
def getFov( N, p0, Nr ):
	conf_str = str(N) + '-' + str(p0) + '-' + str(Nr);

	mappingX = np.load( './fov/' + conf_str + '-mappingX.npy' );
	mappingY = np.load( './fov/' + conf_str + '-mappingY.npy' );

	maxShift = np.load( './fov/' + conf_str + '-maxShift.npy' );
	minShift = -maxShift;

	dims = 2 * ( p0 + Nr ) + 1;
	maxVal = N-1;

	imDims = range( N );
	zdim = dims*dims;
	dim_l = ( np.array( list(range( zdim )) ), );

	abz = np.zeros( (zdim,), dtype='uint32' );

	cache = np.zeros( (213, 213, 3, 81 * 81), dtype='uint32' );
	arr_len = np.zeros( (213,213), dtype='uint32' );
	ind_len = np.zeros( (213,213), dtype='uint32' );

	coords = np.zeros( (N,N,1), dtype='float32' );
	xc, yc = [ N/2., N/2. ];

	for i in range( N ):
		for j in range( N ):
			coords[i,j,:] = np.sqrt( (yc-i)**2 + (xc-j)**2 );

	coords = ((coords - coords.min()) * (1/(coords.max() - coords.min()) * 255)).astype( 'int32' );

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

	def process( im, fovX=0, fovY=0, for_training=True, sample_matrix=False, add_coords=False ):
		fovX = min( max( fovX + minShift, minShift ), maxShift );
		fovY = min( max( fovY + minShift, minShift ), maxShift );

		if sample_matrix:
			orig_reconst = np.zeros_like( im, dtype='uint8' );

		channels = 3;

		if add_coords:
			channels = channels + 1;

		fov = np.zeros( (zdim,channels), dtype='int32' );

		if for_training:
			fov -= 1;

		mX = cache[fovX][fovY][0][:arr_len[fovX][fovY]];
		mY = cache[fovX][fovY][1][:arr_len[fovX][fovY]];
		ab = cache[fovX][fovY][2][:ind_len[fovX][fovY]];

		if sample_matrix:
			orig_reconst[mX,mY,:] = im[mX,mY,:];

		if add_coords:
			im = np.dstack( (im,coords) );

		fov[ab,:] = im[mX,mY,:];
		fov=fov.reshape( (dims,dims,channels) );

		if sample_matrix:
			return [fov,orig_reconst];
		else:
			return fov;

	return process;
