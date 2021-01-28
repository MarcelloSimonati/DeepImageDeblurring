import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Conv2DTranspose, Activation, Add, Concatenate, Lambda, LeakyReLU, BatchNormalization, AveragePooling2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal

#Functional Blocks
def atrous_block(net_in, channels, atrous_conv_kernel, batch_normalization):
    ac_channels = channels//2
    
    ac_1 = Conv2D(ac_channels, atrous_conv_kernel, dilation_rate=1, padding="same", kernel_initializer = HeNormal())(net_in)
    if batch_normalization: ac_1 = BatchNormalization()(ac_1)
    ac_1_act = LeakyReLU()(ac_1)
    
    ac_2 = Conv2D(ac_channels, atrous_conv_kernel, dilation_rate=2, padding="same", kernel_initializer = HeNormal())(net_in)
    if batch_normalization: ac_2 = BatchNormalization()(ac_2)
    ac_2_act = LeakyReLU()(ac_2)
    
    ac_3 = Conv2D(ac_channels, atrous_conv_kernel, dilation_rate=3, padding="same", kernel_initializer = HeNormal())(net_in)
    if batch_normalization: ac_3 = BatchNormalization()(ac_3)
    ac_3_act = LeakyReLU()(ac_3)
    
    ac_4 = Conv2D(ac_channels, atrous_conv_kernel, dilation_rate=4, padding="same", kernel_initializer = HeNormal())(net_in)
    if batch_normalization: ac_4 = BatchNormalization()(ac_4)
    ac_4_act = LeakyReLU()(ac_4)
    
    concat = Concatenate()([ac_1_act, ac_2_act, ac_3_act, ac_4_act])
    conv = Conv2D(channels, atrous_conv_kernel, padding="same", kernel_initializer = HeNormal())(concat)
    if batch_normalization: conv = BatchNormalization()(conv)
    conv_act = LeakyReLU()(conv)
    return Add()([net_in, conv_act])

def res_down_block(net_in, filters, conv_kernel, deconv_kernel, strides):
    conv = Conv2D(2*filters, conv_kernel, strides=strides, padding = "same")(net_in)
    conv_act = LeakyReLU()(conv)
    deconv = Conv2DTranspose(filters, deconv_kernel, strides=strides, padding = "same")(conv_act)
    deconv_act = LeakyReLU()(deconv)
    return Add()([net_in, deconv_act])

def res_up_block(net_in, filters, conv_kernel, deconv_kernel, strides):
    deconv = Conv2DTranspose(int(filters/2), deconv_kernel, strides=strides, padding = "same")(net_in)
    deconv_act = LeakyReLU()(deconv)
    conv = Conv2D(filters, conv_kernel, strides=strides, padding = "same")(deconv_act)
    conv_act = LeakyReLU()(conv)
    return Add()([net_in, conv_act])
    
def retrieve_middle_frame(x):
    return  x[:, :, :, 3:6]

def local_conv(kernel_size, img, kernel_2d):
    # img (B, H, W, 3)
    # kernel_2d (B, H, W, k*k)

    # Convolve with kernel_2d
    k = kernel_size
    _, h, w, c = tf.unstack(tf.shape(img))
    result = tf.image.extract_patches(img, sizes=(1,k,k,1), strides=(1,1,1,1),rates=(1,1,1,1), padding="SAME") # Output [B, H, W, k*k*c]
    result = tf.reshape(result,[-1, h, w, k*k, c]) # Output [B, H, W, k*k, c]
    kernel_2d = tf.expand_dims(kernel_2d, axis=-1) # (B, H, W, k*k, 1). Because of the RGB dimension
    result = tf.multiply(result,kernel_2d) # Elementwise multiplication. Resulting (B, H, W, k*k, 3)
    result = tf.reduce_sum(result,axis=3) # (B, H, W, 3)

    return result

#Model Creation Functions
def get_atrous_net(mode = "image", 
                  image_size = (320,320), 
                  atrous_blocks = 20, 
                  feature_extraction_kernel = (9,9), 
                  standard_kernel = (3,3), 
                  up_down_sample_stride = 2, 
                  atrous_conv_kernel = (3,3), 
                  atrous_channels = 256,
                  batch_normalization = False):

    if mode == "image":
        input_shape = (image_size[0], image_size[1], 3)
    elif mode == "video":
        input_shape = (image_size[0], image_size[1], 9)
    else: raise Exception('ERROR: "{}" is not a supported mode. Supported modes are "image" and "video"'.format(mode))

    inputs = Input(shape=input_shape)
    to_sum = inputs

    conv_1 = Conv2D(atrous_channels//2, feature_extraction_kernel, padding="same", kernel_initializer = HeNormal())(inputs)
    if batch_normalization: conv_1 = BatchNormalization()(conv_1)
    conv_1_act = LeakyReLU()(conv_1)

    conv_2 = Conv2D(atrous_channels, standard_kernel, strides = up_down_sample_stride, padding="same", kernel_initializer = HeNormal())(conv_1_act)
    if batch_normalization: conv_2 = BatchNormalization()(conv_2)
    conv_2_act = LeakyReLU()(conv_2)

    atr = conv_2_act
    for i in range(atrous_blocks):
        atr = atrous_block(atr, atrous_channels, atrous_conv_kernel, batch_normalization)

    deconv = Conv2DTranspose(atrous_channels//2, standard_kernel, strides = up_down_sample_stride, padding="same",kernel_initializer = HeNormal())(atr)
    if batch_normalization: deconv = BatchNormalization()(deconv)
    deconv_act = LeakyReLU()(deconv)
    
    concat = Concatenate()([deconv_act, conv_1_act])
    
    conv_3 = Conv2D(atrous_channels//4, standard_kernel, padding="same", kernel_initializer = HeNormal())(concat)
    if batch_normalization: conv_3 = BatchNormalization()(conv_3)
    conv_3_act = LeakyReLU()(conv_3)
    
    conv_4_act = Conv2D(3, standard_kernel, padding="same", activation="linear", kernel_initializer = HeNormal())(conv_3_act)
    
    if input_shape[-1] == 9:
        to_sum = Lambda(retrieve_middle_frame)(inputs)
       
    out = Add()([to_sum, conv_4_act])


    return Model(inputs=inputs, outputs=[out])

def get_carlo_net(n_filter = 128, kernel_size = (3,3)):
    inputs = Input(shape=(None,None, 3)) 
    output = Conv2D(n_filter, kernel_size, activation='relu')(inputs) #TO SKIP
    output_1 = Conv2D(n_filter, kernel_size, activation='relu')(output) 
    output_2 = Conv2D(n_filter, kernel_size, activation='relu')(output_1) #TO SKIP
    output_3 = Conv2D(n_filter, kernel_size, activation='relu')(output_2) 
    output_4 = Conv2D(n_filter, kernel_size, activation='relu')(output_3) #TO SKIP
    output_5 = Conv2D(n_filter, kernel_size, activation='relu')(output_4) 
    output_6 = Conv2D(n_filter, kernel_size, activation='relu')(output_5) #TO SKIP
    output_7 = Conv2D(n_filter, kernel_size, activation='relu')(output_6) 
    output_8 = Conv2D(n_filter, kernel_size, activation='relu')(output_7) #TO SKIP 
    output_9 = Conv2D(n_filter, kernel_size, activation='relu')(output_8) 

    output_10 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_9)
    output_11 = Add()([output_8, output_10])
    output_12 = Activation('relu')(output_11)
    output_13 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_12)
    output_14 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_13)
    output_15 = Add()([output_6, output_14])
    output_16 = Activation('relu')(output_15)
    output_17 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_16)
    output_18 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_17)
    output_19 = Add()([output_4, output_18])
    output_20 = Activation('relu')(output_19)
    output_21 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_20)
    output_22 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_21)
    output_23 = Add()([output_2, output_22])
    output_24 = Activation('relu')(output_23)
    output_25 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_24)
    output_26 = Conv2DTranspose(n_filter, kernel_size, activation='relu')(output_25)
    output_27 = Add()([output, output_26])
    output_28 = Activation('relu')(output_27)
    output_29 = Conv2DTranspose(3, kernel_size, activation='relu')(output_28)

    return Model(inputs=inputs, outputs=[output_29])

def get_kaist_net(mode = 'video',
                    image_size = (256, 256),
                    conv_kernel = (5,5),
                    deconv_kernel = (4,4),
                    strides = 2,
                    res_blocks_1 = 9,
                    res_blocks_2 = 4,
                    k = 5):
    
    if mode == "image":
        input_shape = (image_size[0], image_size[1], 3)
    elif mode == "video":
        input_shape = (image_size[0], image_size[1], 9)
    else: raise Exception('ERROR: "{}" is not a supported mode. Supported modes are "image" and "video"'.format(mode))
    
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(32, conv_kernel, padding = "same")(inputs)
    conv_1_act = LeakyReLU()(conv_1)

    rdu = conv_1_act
    for res in range(res_blocks_1):
        rdu = res_down_block(rdu, 32, conv_kernel, deconv_kernel, strides)
    shortcut_1 = rdu
    avg_pool_1 = AveragePooling2D()(rdu)
    conv_1x1_1 = Conv2D(64, (1,1), padding = "same")(avg_pool_1)

    rdu = conv_1x1_1
    for res in range(res_blocks_1):
        rdu = res_down_block(rdu, 64, conv_kernel, deconv_kernel, strides)

    shortcut_2 = rdu

    avg_pool_2 = AveragePooling2D()(rdu)
    conv_1x1_2 = Conv2D(128, (1,1), padding = "same")(avg_pool_2)

    rdu = conv_1x1_2
    for res in range(res_blocks_2):
        rdu = res_down_block(rdu, 128, conv_kernel, deconv_kernel, strides)
    rud = rdu
    for res in range(res_blocks_2):
        rud = res_up_block(rud, 128, conv_kernel, deconv_kernel, strides)
        
    up_sample_1 = UpSampling2D() (rud)
    concat_1 = Concatenate()([up_sample_1, shortcut_2])
    conv_1x1_3 = Conv2D(64, (1,1), padding = "same")(concat_1)

    rud = conv_1x1_3
    for res in range(res_blocks_1):
        rud = res_up_block(rud, 64, conv_kernel, deconv_kernel, strides)
        
    residual_output = rud

    #First branch

    conv_2 = Conv2D(64, conv_kernel, padding = "same")(residual_output)
    conv_2_act = LeakyReLU()(conv_2)
    up_sample_2 = UpSampling2D() (conv_2_act)
    concat_2 = Concatenate()([up_sample_2, shortcut_1])
    conv_1x1_4 = Conv2D(32, (1,1), padding = "same")(concat_2)
    conv_3 = Conv2D(32, conv_kernel, padding = "same")(conv_1x1_4)
    conv_3_act = LeakyReLU()(conv_3)

    rgb = Conv2D(3, conv_kernel, padding = "same")(conv_3_act)
    w = Conv2D(1, conv_kernel, padding = "same", activation="sigmoid")(conv_3_act)

    #Second branch
    conv_6 = Conv2D(64, conv_kernel, padding = "same")(residual_output)
    conv_6_act = LeakyReLU()(conv_6)
    up_sample_3 = UpSampling2D() (conv_6_act)
    conv_7 = Conv2D(32, conv_kernel, padding = "same")(up_sample_3)
    k2d = Conv2D(k*k, conv_kernel, padding = "same")(conv_7)
    
    if mode == "image":
        middle_frame = inputs
    elif mode == "video":
        middle_frame = Lambda(retrieve_middle_frame)(inputs)
    else: raise Exception('ERROR: "{}" is not a supported mode. Supported modes are "image" and "video"'.format(mode))

    output_k2d = local_conv(k, middle_frame, k2d)

    output = w*output_k2d + (1-w)*rgb 

    return Model(inputs=inputs, outputs=[output])