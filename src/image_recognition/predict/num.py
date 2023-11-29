import os
import json
import torch
import torchvision

from glob import glob
from image_recognition.model.NET import ModelMNIST
from image_recognition import transforms


# Getting the path to the model
with open(os.path.abspath("./scripts/destination_path.json")) as f:
    destination = json.load(f)
path_model = glob(os.path.abspath("./%s/*.pth") % destination['path_model_load'])[-1]

# Getting the configuration
with open(os.path.abspath("./scripts/training_config.json")) as f:
   training_config = json.load(f)

class PredictImage():
    """
    The class is only used to predict handwritten numbers.

    Args:
    path (string, optional): The path to load the model.
    config (dictionary, optional): Hyperparameter configuration defining the model.
    transformation (NoneType, optional): A function/transform that  takes in an PIL image
             and returns a transformed version. E.g, ``transforms.PilToTensor()``

    Attributes:
    load_model: Loads the model at the specified path.
    redict: Predicts the image.

    """
    def __init__(self, path: str = path_model, config: dict = training_config, transformation = None):
        self.path = path
        self.training_config = config

        if transformation:
            self.transforms = transformation
        else:
            self.transforms = transforms.dict_transforms['val']


    def __call__(self, id: str) -> str:
        # path of the PNG image.
        img_path = os.path.abspath("./data/forward/%s.png") % id
        # reading an image
        img_tensor = torchvision.io.read_image(img_path)
        # performing transformations on the PIL image
        img = self.transforms(img_tensor)
        # add the "batch" dimension to get the form Tensor[ "batch", channels, height, width
        img = img.unsqueeze(0)

        model = self.load_model()
        out = self.predict(img, model).item()
        if out is not None:
            # delete the image from the disk
            os.remove(img_path)
    
        return str(out)


    def load_model(self):
        # Initializing the model
        model = ModelMNIST(
            hidden_size = self.training_config['hidden_size'],
            dropout_prob = self.training_config['dropout_prob'],
            ).to("cpu")

        # loading the model checkpoint
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model


    def predict(self, img, model, device: str = 'cpu') -> int:
        model.eval()
        with torch.no_grad():
            inputs = img.to(device)
            outputs = model(inputs)

            predicted = torch.max(outputs.data, 1)[1]

        return predicted
