
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import time
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import linear_model

#testimgpath = "image_path"
#testimgpath = "/afs/crc.nd.edu/user/p/psharma3/NNproject/images/sample_image.jpg"
#testimgpath = "../images/sample_image.jpg"

testimgpath = "../images/new_google_test_images/angry_image.jpeg"


#saved_model_path = "../trained_models/resnet50_model_final_kag.pth"
#saved_model_path = "../saved_models/SimpleNet.pt"

saved_model_path = "../new_saved_models/new.pt"


def test(test_imgpath,saved_model_path):
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5069, 0.4518, 0.4377], std=[0.2684, 0.2402, 0.2336])])
    
    # check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load models

    ##### conv pytorch for Resnet ######
    #import torchvision.models as models
    #model = models.resnet18()

    ## LeNet ########
    #model = conv_model.LeNet(3,7)

    ## MLP ###
    #model = linear_model.SimpleNet()  


    # intializing the loss functions and optimizer
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(SimpleNet.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Loading the pre-trained weights......")

    
    model= torch.load(saved_model_path)

    model.eval()
    model.to(device)

    print("Loading the image and resizing")
    image = Image.open(test_imgpath)
    image = image.resize((32, 32)).convert("RGB")
    #image = image.resize((224, 224)).convert("RGB")
    
    #image = image.convert("RGB")
    convert_tensor = transforms.ToTensor()
    
    image = convert_tensor(image).to(device)
    image = image.unsqueeze(0) # Add batch dimension

    print(image.shape)
    
    
    print("model prediction!")
    outputs = model(image)

                         
    _,pred = torch.max(outputs, 1)

    print(f"The predicted label is {pred}")

    if pred == 0:
        pred = "angry"
    elif pred == 1:
        pred = "disguist"
    elif pred == 2:
        pred = "fear"
    elif pred == 3:
        pred = "happy"
    elif pred == 4:
        pred = "neutral"
    elif pred == 5:
        pred = "sad"
    elif pred == 6:
        pred = "surprise"

    print("#"*20)
    print(f"The model predicted the image to be {pred}")
    print("#"*20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a test on ResNet-18 trained model")
    parser.add_argument("--testimgpath", "-t", required = True, help="test image path")
    parser.add_argument("--modelpath", "-p", required = True, help="modelpath")
  
    args = parser.parse_args()
    
    test(test_imgpath=args.testimgpath,saved_model_path=args.modelpath)
