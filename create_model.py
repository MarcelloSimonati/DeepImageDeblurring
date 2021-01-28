import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from argparse import ArgumentParser
from Models import get_carlo_net, get_kaist_net, get_atrous_net

parser = ArgumentParser()
parser.add_argument("-d", "--directory", 
                    dest="directory", 
                    required=True,
                    help="Name of the directory where to save the model. If not present it will create a new directory in the specified path",
                    metavar="DIR")

parser.add_argument("-m", "--model", 
                    dest="model", 
                    required=True, 
                    choices=['carlo_net', 'kaist_net', 'atrous_net'],
                    help="Name of the model", metavar="MODEL")

parser.add_argument("-is", "--image_size", 
                    dest="image_size",
                    nargs="+",
                    type=int, 
                    required=False,
                    help="Size of the images in the training directory",
                    metavar="H W")

parser.add_argument("-mode", 
                    dest="mode", 
                    required=False,
                    choices=['image', 'video'],
                    help="Select mode to create the model. With image data or a custom video-frame directory",
                    metavar="MODE")    

args = vars(parser.parse_args())

if not os.path.isdir(args["directory"]):
    os.mkdir(args["directory"])

filepath = os.path.join(args["directory"], "model.json")

if args["model"] == "carlo_net":
    if args['mode'] == 'video':
        print("ERROR: Cannot create carlo_net in video mode")
        exit()
    model = get_carlo_net()
    model_json = model.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    model.summary()
    print("Saved model to: {}".format(filepath))


elif args["model"] == "kaist_net":
    if args['image_size'] == None or args['mode'] == None:
        print("ERROR: You should provide mode and image size for kaist_net")
        exit()
    model = get_kaist_net(image_size=(args['image_size'][0], args['image_size'][1]), mode=args['mode'])
    model_json = model.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    model.summary()
    print("Saved model to: {}".format(filepath))

elif args["model"] == "atrous_net":
    if args['image_size'] == None or args['mode'] == None:
        print("ERROR: You should provide mode and image size for atrous_net")
        exit()
    model = get_atrous_net(image_size=(args['image_size'][0], args['image_size'][1]), mode=args['mode'])
    model_json = model.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    model.summary()
    print("Saved model to: {}".format(filepath))