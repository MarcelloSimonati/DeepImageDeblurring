import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from argparse import ArgumentParser
import tensorflow as tf
from CustomMetrics import *
from CifarDatagen import cifar_datagen
from VideoDatagen import VideoDatagen
from datetime import datetime
import pandas as pd
now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")

parser = ArgumentParser()
parser.add_argument("-md", "--model-directory", 
                    dest="modeldir", 
                    required=True,
                    help="Name of the directory where the model is saved. If model.json is not present in the directory, please create a model using the create_model.py script",
                    metavar="DIR")

parser.add_argument("-mode", 
                    dest="mode", 
                    required=True,
                    choices=['cifar', 'video'],
                    help="Select mode to run the model. With CIFAR10 data or a custom video-frame directory",
                    metavar="MODE")    

parser.add_argument("-dd", "--data-directory", 
                    dest="datadir", 
                    required=False,
                    help="Name of the directory where the training data is stored for video mode. See the README to know how to structure the data directory",
                    metavar="DIR")

parser.add_argument("-is", "--image_size", 
                    dest="image_size",
                    nargs="+",
                    type=int, 
                    required=False,
                    help="Size of the images in the training directory",
                    metavar="H W")

parser.add_argument("-ts", "--target_size", 
                    dest="target_size",
                    nargs="+",
                    type=int, 
                    required=False,
                    help="Size of the target size for the patches of the images in the training directory",
                    metavar="H W")

parser.add_argument("-lr", "--learning_rate",
                    type=float,
                    dest="learning_rate", 
                    required=True,
                    help="Learning rate of the optimizer",
                    metavar="XX")

parser.add_argument("-bs", "--batch_size",
                    type=int,
                    dest="batch_size", 
                    required=True,
                    help="Batch size for the fit function",
                    metavar="XX")

parser.add_argument("-e", "--epochs",
                    type=int,
                    dest="epochs", 
                    required=True,
                    help="Number of training epochs",
                    metavar="XX")

parser.add_argument("-wf", "--weights_file", 
                    dest="weights", 
                    required=False,
                    help=".h5 file where the model weights are saved. To be found inside the specified directory", metavar="weightfile.h5")
                     
parser.add_argument("-s", "--seed", 
                    dest="seed",
                    type=int,
                    default=42, 
                    required=False,
                    help="Seed used in the data generators", metavar="XX")
                    
parser.add_argument("-w", "--workers", 
                    dest="workers",
                    type=int,
                    default=1, 
                    required=False,
                    help="Number of workers to use in the fit function", metavar="XX")

args = vars(parser.parse_args())
if args['mode'] == 'video' and (args['datadir'] == None or args['image_size'] == None) or args['target_size'] == None:
    print("ERROR: To use the video mode you need to specify a correct data directory and image size")
    exit()

print("### LOADING MODEL ###")
json_file = open(os.path.join(args['modeldir'], 'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

if args['weights'] != None:
    print("### LOADING WEIGHTS ###")
    model.load_weights(os.path.join(args['modeldir'], args['weights']))
    
print("### COMPILING MODEL ###")
opt = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'], clipnorm = True)
model.compile(optimizer=opt, loss=lad_loss, metrics=[ssim_metric, psnr_metric, 'mse'])


if args['mode'] == "video":
    print("### LOADING DATA ###")
    datagen = VideoDatagen((args["image_size"][0],args["image_size"][1]), (args["target_size"][0],args["target_size"][1]), args['batch_size'], args['seed'], data_dir=args['datadir'])
    X = datagen.train_generator()
    val_data = datagen.val_generator()
    print("")
    print("### TRAINING MODEL###")
    print("")
    history = model.fit(X, 
            validation_data = val_data, 
            epochs = args['epochs'],
            steps_per_epoch = datagen.train_samples/(3*args['batch_size']), 
            validation_steps = datagen.val_samples/(3*args['batch_size']),
            workers=args['workers'])
    print("### TRAINING ENDED ###")
    print("")
    print("### SAVING MODEL ###")
    model.save_weights(os.path.join(args['modeldir'],'weights-{}.h5'.format(now)))
    print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
    print("")
    print("### SAVING HISTORY ###")
    df_hist = pd.DataFrame.from_dict(history.history)
    df_hist.to_csv(os.path.join(args['modeldir'],'history-{}.csv'.format(now)), mode='w', header=True)
    print("Weights saved to: history-{}.csv inside the model directory".format(now))
    print("")

elif args['mode'] == 'cifar':
    print("### LOADING DATA ###")
    (x_train, x_train_blur), (x_test, x_test_blur) = cifar_datagen(random_seed=args['seed'])
    print("")
    print("### TRAINING MODEL###")
    print("")
    history = model.fit(x = x_train_blur,
            y = x_train,
            batch_size =  args['batch_size'],
            validation_split = 0.2, 
            epochs= args['epochs'],
            workers = args['workers'])
    print("### TRAINING ENDED ###")
    print("")
    print("### SAVING MODEL ###")
    model.save_weights(os.path.join(args['modeldir'],'weights-{}.h5'.format(now)))
    print("Weights saved to: weights-{}.h5 inside the model directory".format(now))
    print("")
    print("### SAVING HISTORY ###")
    df_hist = pd.DataFrame.from_dict(history.history)
    df_hist.to_csv(os.path.join(args['modeldir'],'history-{}.csv'.format(now)), mode='w', header=True)
    print("Weights saved to: history-{}.csv inside the model directory".format(now))
    print("")
                    