from tensorflow.keras import layers, models, optimizers, regularizers, losses;
import numpy as np;

from tensorflow.compat.v2.keras.utils import multi_gpu_model;
from tensorflow.keras.utils import plot_model;
import tensorflow as tf;

from tensorflow.python.keras.utils import losses_utils

from tensorflow.keras import backend as K;

class AttMapMetric( tf.keras.metrics.Metric ):
    def __init__( self, name='att_map', layer=None, shape=(10,10), normalizer=80./9., **kwargs ):
        super(AttMapMetric, self).__init__( name=name, **kwargs );
        self.layer_to_attmap = layer;
        self.normalizer = normalizer;
        self.sh_flatten = (np.prod( shape ),);
        self.sh = shape;

    def update_state( self, y_true, y_pred, sample_weight=None ):
        pass;

    def result( self ):
        inp = K.mean( self.layer_to_attmap, axis=-1, keepdims=True );
        x = tf.squeeze(inp, axis=-1);
        x = layers.Reshape( self.sh_flatten )(x);
        x = tf.argmax( x, axis=-1 );
        x = tf.unravel_index( x, self.sh );
        return tf.cast( tf.cast(x,dtype=tf.float32)*self.normalizer, dtype=tf.uint8 );

    def reset_states( self ):
        pass;

def build_xception_core50( batch_size=None, add_coords = False, add_dist = False, additional_blocks=8, learning_rate=0.045 ):
    tf.compat.v1.disable_eager_execution();

    input_channels = 3;

    if add_coords:
        input_channels += 1;

    if add_dist:
        input_channels += 1;

    shape = (81,81,input_channels);

    channel_axis = -1;

    img_input = layers.Input( shape=shape, batch_size=batch_size, name='img_input' );

    x = layers.Conv2D( 32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1' )( img_input );
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x);
    x = layers.Activation('relu', name='block1_conv1_act')(x);
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x);
    x = layers.Activation('relu', name='block1_conv2_act')(x);

    residual = layers.Conv2D(64, (1, 1), padding='same', use_bias=False, name='conv2d')(x);
    residual = layers.BatchNormalization(axis=channel_axis,name='batch_normalization')(residual);

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block2_sepconv2_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x);

    x = layers.add([x, residual]);

    residual = layers.Conv2D(64, (1, 1), padding='same', use_bias=False, name='conv2d_1')(x);
    residual = layers.BatchNormalization(axis=channel_axis,name='batch_normalization_1')(residual);

    x = layers.Activation('relu', name='block3_sepconv1_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block3_sepconv2_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x);

    x = layers.add([x, residual]);

    residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='conv2d_2')(x);
    residual = layers.BatchNormalization(axis=channel_axis,name='batch_normalization_2')(residual);

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block4_sepconv2_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x);

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x);
    x = layers.add([x, residual]);

    for i in range(additional_blocks):
        residual = x;
        prefix = 'block' + str(i + 5);

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x);
        x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x);
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x);
        x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x);
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x);
        x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x);

        x = layers.add([x, residual]);

    residual = layers.Conv2D(64, (1, 1), padding='same', use_bias=False, name='conv2d_3')(x);
    residual = layers.BatchNormalization(axis=channel_axis,name='batch_normalization_3')(residual);

    x = layers.Activation('relu', name='block13_sepconv1_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block13_sepconv2_act')(x);
    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x);

    x = layers.add([x, residual]);

    x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block14_sepconv1_act')(x);

    actual_last_conv = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(actual_last_conv);
    last_conv = layers.Activation('relu', name='last_conv')(x);

    x = layers.GlobalAveragePooling2D( name='avg_pool' )(last_conv);

    x = layers.Dropout(0.8)(x);
    x = layers.Dense( 50, activation='softmax', name='predictions' )(x);

    model = models.Model( inputs=[img_input], outputs=[x], name='xception' );

    alpha = 0.00001;

    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            layer.add_loss(regularizers.l2(alpha)(layer.kernel));

        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(regularizers.l2(alpha)(layer.bias));

    #model = multi_gpu_model( model, gpus=2 );

    optimizer = optimizers.SGD(
        lr=learning_rate,
        momentum=0.9
    );

    att_map_metric = AttMapMetric( name='att_map', layer=model.get_layer( 'last_conv' ).output, shape=(19,19), normalizer=80./18. );

    loss = { 'predictions': 'categorical_crossentropy' };
    metrics = { 'predictions': ['acc',att_map_metric] };

    model.compile( loss=loss, metrics=metrics, optimizer=optimizer );

    model.summary();

    return model;

def build_xception_imagenet(batch_size):
    tf.compat.v1.disable_eager_execution();

    img_input = layers.Input( shape=(81,81,3), batch_size=batch_size );

    channel_axis = -1;

    x = layers.Conv2D( 32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1' )( img_input );
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x);
    x = layers.Activation('relu', name='block1_conv1_act')(x);
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x);
    x = layers.Activation('relu', name='block1_conv2_act')(x);

    residual = layers.Conv2D(128, (1, 1), padding='same', use_bias=False)(x);
    residual = layers.BatchNormalization(axis=channel_axis)(residual);

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block2_sepconv2_act')(x);
    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x);

    x = layers.add([x, residual]);

    residual = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x);
    residual = layers.BatchNormalization(axis=channel_axis)(residual);

    x = layers.Activation('relu', name='block3_sepconv1_act')(x);
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block3_sepconv2_act')(x);
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x);

    x = layers.add([x, residual]);

    residual = layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x);
    residual = layers.BatchNormalization(axis=channel_axis)(residual);

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block4_sepconv2_act')(x);
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x);

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x);
    x = layers.add([x, residual]);

    for i in range(8):
        residual = x;
        prefix = 'block' + str(i + 5);

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x);
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x);
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x);
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x);
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x);
        x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x);
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x);

        x = layers.add([x, residual]);

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x);
    residual = layers.BatchNormalization(axis=channel_axis)(residual);

    x = layers.Activation('relu', name='block13_sepconv1_act')(x);
    x = layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block13_sepconv2_act')(x);
    x = layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x);

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x);
    x = layers.add([x, residual]);

    x = layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x);
    x = layers.Activation('relu', name='block14_sepconv1_act')(x);

    x = layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x);
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x);
    last_conv = layers.Activation('relu', name='last_conv')(x);

    x = layers.GlobalAveragePooling2D( name='avg_pool' )(last_conv);
    #x = layers.Dropout(0.6)(x);
    x = layers.Dense( 1000, activation='softmax', name='predictions' )(x);

    model = models.Model( inputs=[img_input], outputs=[x,last_conv], name='xception' );

    alpha = 0.00001;

    """
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            layer.add_loss(regularizers.l2(alpha)(layer.kernel));

        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(regularizers.l2(alpha)(layer.bias));
    """

    #model = multi_gpu_model( model, gpus=2 );

    optimizer = optimizers.SGD(
        lr=0.045,
        momentum=0.9
    );

    def my_loss( t, p ):
        return p;

    class LossFunctionWrapper(losses.Loss):
        def __init__(self,
            fn,
            reduction=losses_utils.ReductionV2.NONE,
            name=None,
            **kwargs):
            super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
            self.fn = fn
            self._fn_kwargs = kwargs

        def call(self, y_true, y_pred):
            return self.fn(y_true, y_pred, **self._fn_kwargs)

        def get_config(self):
            config = {}
            for k, v in six.iteritems(self._fn_kwargs):
                config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
            base_config = super(LossFunctionWrapper, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class MyLoss(LossFunctionWrapper):
        def __init__(self,
               reduction=losses_utils.ReductionV2.NONE,
               name='my_loss'):
            super(MyLoss, self).__init__(
                my_loss,
                name=name,
                reduction=reduction)

    loss = { 'predictions': tf.keras.losses.CategoricalCrossentropy(reduction=losses_utils.ReductionV2.AUTO), 'last_conv': MyLoss() };
    metrics = { 'predictions': ['acc'], 'last_conv': [] };

    model.compile( loss=loss, metrics=metrics, optimizer=optimizer );

    model.summary();
    print( model.metrics_names );

    return model;
