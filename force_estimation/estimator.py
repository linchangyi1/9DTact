import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import yaml
import torch
import numpy as np
np.set_printoptions(suppress=True)
import argparse
from model import *
import cv2


class Estimator:
    def __init__(self, cfg):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", default=None, type=str)
        parser.add_argument("--model_layer", default=None, type=str)
        parser.add_argument("--cuda_index", default=None, type=int)
        args = parser.parse_args()

        # parameters
        model_name = args.model_name if args.model_name is not None else cfg['model_list'][cfg['model_choice']][0]
        model_layer = args.model_layer if args.model_name is not None else cfg['model_list'][cfg['model_choice']][1]
        model_type = model_name + '-' + str(model_layer)
        cuda_index = args.cuda_index if args.cuda_index is not None else cfg['cuda_index']

        # model configuration
        self.device = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
        if model_name == 'Resnet':
            self.model = Resnet(layer=int(model_layer), pretrained=False).to(self.device)
        elif model_name == 'Densenet':
            self.model = Densenet(layer=int(model_layer), pretrained=False).to(self.device)

        self.save_dir = cfg['save_dir'] + '/' + model_type + str(cfg['weights'][cfg['model_choice']][0])
        weights_path = self.save_dir + '/epoch_' + str(cfg['weights'][cfg['model_choice']][1]) + '.pt'
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        # inferring the first image takes more time than the latter images, so we pass it
        with torch.no_grad():
            two_img = np.ones((2, 3, cfg['img_size'][0], cfg['img_size'][1]))
            representation = torch.from_numpy(two_img).float().to(self.device)
            self.model(representation)
        print("Ready to predict the 6D force!")

    def predict_force(self, representation):
        representation = representation.transpose([2, 0, 1])
        # Since we find some versions of pytorch output random result if the batch_size is 1,
        # we then copy the input image to set the batch_size to 2.
        # If the version of pytorch that you are using has no the mentioned issue, then you can set it to 1.
        input_batch = 2
        if input_batch == 2:
            two_img = np.array([representation, representation])
            representation = torch.from_numpy(two_img).float().to(self.device)
            with torch.no_grad():
                force = self.model(representation)
                force[force < 0] = 0
                force[force > 1] = 1
                force = force.cpu().numpy()[0]
        else:
            representation = torch.from_numpy(np.array(representation)).float().to(self.device)
            with torch.no_grad():
                force = self.model(representation)
                force[force < 0] = 0
                force[force > 1] = 1
                force = force.cpu().numpy()
        return force


if __name__ == '__main__':
    f = open("force_config.yaml", 'r+', encoding='utf-8')
    config = yaml.load(f, Loader=yaml.FullLoader)
    estimator = Estimator(config)
    image = cv2.imread('test.png')
    input_image = np.zeros_like(image)
    input_image[::, ::, 0] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    predicted_force = estimator.predict_force(input_image)
    print(predicted_force)



