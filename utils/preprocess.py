import numpy as np, cv2, os, random, json, pickle;
from os.path import isfile;

def printProgressBar(iteration, total, prefix = 'Progress:', suffix = '', decimals = 1, length = 25, fill = '▒', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\r%s ▕%s▏ %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)

def printLoader( model ):
    states = [ '[x         ]', '[ x        ]', '[  x       ]', '[   x      ]', '[    x     ]', '[     x    ]', '[      x   ]', '[       x  ]', '[        x ]', '[         x]' ];
    l = len( states );
    step = 0;

    def draw( message ):
        nonlocal step;
        if step % l == 0:
            step = 0;

        print( '\rModel: ' + model + ' ' + states[step%l] + ' # ' + message + '' + ( ' '*100 ), end = '\r' );

        step += 1;

    return draw;
