from PIL import Image;
import os, json;
import warnings;
warnings.filterwarnings("error");

source = './val_raw/';
target = './val/';
removed = 0;

for dirpath, dirs, files in os.walk(source):
    max = len(files);
    co = 0;

    for filename in files:
        fname = os.path.join( dirpath, filename );

        if co % 1000 == 0:
            print( str(co) + '/' + str(max) );
        co += 1;

        try:
            im = Image.open( fname );

            w, h = im.size;

            if w < h:
                nw = 256;
                nh = int(h*nw/w);
                l = 0;
                r = 256;
                t = int( (nh-256)/2 );
                b = int( (nh-256)/2 ) + 256;
            else:
                nh = 256;
                nw = int(w*nh/h);
                t = 0;
                b = 256;
                l = int( (nw-256)/2 );
                r = int( (nw-256)/2 ) + 256;

            im = im.resize( (nw,nh), Image.BICUBIC );
            im = im.crop( (l, t, r, b) );

            im.save( target + filename );
        except:
            os.remove( target + filename );
            removed += 1;
            pass;

print( removed );
