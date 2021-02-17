import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from argparse import ArgumentParser
import tensorflow as tf
from CustomMetrics import *
from CifarDatagen import cifar_datagen
from VideoDatagen import VideoDatagen
import matplotlib.pyplot as plt
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

parser.add_argument("-wf", "--weights_file", 
                    dest="weights", 
                    required=True,
                    help=".h5 file where the model weights are saved. To be found inside the specified directory", metavar="weightfile.h5")

parser.add_argument("-mode", 
                    dest="mode", 
                    required=True,
                    choices=['cifar', 'video'],
                    help="Select mode to test the model. With CIFAR10 data or a custom video-frame directory",
                    metavar="MODE") 

parser.add_argument("-test_mode", 
                    dest="test_mode", 
                    required=True,
                    choices=['batch', 'evaluate'],
                    help="Select mode to test the model. Mode batch will create a RESULTS directory inside the model with the predicted images, evaluate will run a full evaluation on the test set",
                    metavar="MODE")

parser.add_argument("-n_batch",
                    type=int,
                    dest="n_batch",
                    default=1, 
                    required=False,
                    help="Number of batches of prediction to make",
                    metavar="XX")     

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

parser.add_argument("-bs", "--batch_size",
                    type=int,
                    dest="batch_size", 
                    required=True,
                    help="Batch size for the fit function",
                    metavar="XX")

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

if args['test_mode'] == 'batch' and args['n_batch'] == None:
    print("ERROR: To use the batch mode you need to specify a correct number of batches")
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
opt = tf.keras.optimizers.Adam(learning_rate=0.0, clipnorm = True)
model.compile(optimizer=opt, loss=lad_loss, metrics=[ssim_metric, psnr_metric, 'mse'])

if args['mode'] == "video":
    print("### LOADING DATA ###")
    datagen = VideoDatagen((args["image_size"][0],args["image_size"][1]), (args["target_size"][0],args["target_size"][1]), args['batch_size'], args['seed'], data_dir=args['datadir'])
    test_data = datagen.test_generator()

    if args['test_mode'] == 'evaluate':
        print("")
        print("### EVALUATING MODEL ###")
        print("")
        history = model.evaluate(test_data,
                steps = datagen.test_samples/(3*args['batch_size']),
                workers=args['workers'])
        print("### EVALUATION ENDED ###")
        print("")
        print("### SAVING RESULTS ###")
        df_hist = pd.DataFrame(history, index = ['loss', 'ssim', 'psnr', 'mse'], columns =['Values'])
        df_hist.to_csv(os.path.join(args['modeldir'],'results-{}.csv'.format(now)), mode='w', header=True)
    
    elif args['test_mode'] == 'batch':
        print("")
        print("### CREATING TEST IMAGES ###")
        print("")
        img_dir = os.path.join(args["modeldir"], "TEST_IMGS_{}".format(now))
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        j = 0
        i = 0
        n = 0
        for j in range(args['n_batch']):
            test_imgs = next(test_data)
            predicted = model.predict(test_imgs[0])
            for i in range(args['batch_size']):
                plt.figure(figsize=(21,7))
                plt.subplot(131)
                plt.imshow(test_imgs[0][i,:,:,3:6])
                plt.subplot(132)
                plt.imshow(predicted[i])
                plt.subplot(133)
                plt.imshow(test_imgs[1][i])
                plt.savefig(os.path.join(img_dir,"img{}.png".format(n)))
                n+=1
        print("Figures saved to {}".format(img_dir))

elif args['mode'] == 'cifar':
    print("### LOADING DATA ###")
    (x_train, x_train_blur), (x_test, x_test_blur) = cifar_datagen(random_seed=args['seed'])

    if args['test_mode'] == 'evaluate':
        print("")
        print("### EVALUATING MODEL###")
        print("")
        history = model.evaluate(x = x_test_blur, y = x_test, batch_size = args['batch_size'], workers=args['workers'])
        print("### EVALUATION ENDED ###")
        print("")
        print("### SAVING RESULTS ###")
        df_hist = pd.DataFrame(history, index = ['loss', 'ssim', 'psnr', 'mse'], columns =['Values'])
        df_hist.to_csv(os.path.join(args['modeldir'],'results-{}.csv'.format(now)), mode='w', header=True)
    
    elif args['test_mode'] == 'batch':
        print("")
        print("### CREATING TEST IMAGES ###")
        print("")
        img_dir = os.path.join(args["modeldir"], "TEST_IMGS_{}".format(now))
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        j = 0
        i = 0
        n = 0
        for j in range(args['n_batch']):
            blur_imgs = x_test_blur[j*args['batch_size']:(j+1)*args['batch_size']]
            sharp_imgs = x_test[j*args['batch_size']:(j+1)*args['batch_size']]
            predicted = model.predict(blur_imgs, workers = args['workers'], batch_size = args['batch_size'])
            for i in range(args['batch_size']):
                plt.figure(figsize=(21,7))
                plt.subplot(131)
                plt.imshow(blur_imgs[i])
                plt.subplot(132)
                plt.imshow(predicted[i])
                plt.subplot(133)
                plt.imshow(sharp_imgs[i])
                plt.savefig(os.path.join(img_dir,"img{}.png".format(n)))
                n+=1
        print("Figures saved to {}".format(img_dir))