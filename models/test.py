import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the transform to apply to the input image
# defining the transforms and dataloaders
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the model
model = torch.load("../saved_models/Resnet.pt", map_location=torch.device('cpu'))
model.eval()
# Load the input image

test_imgs_path = np.loadtxt("../test_images.txt",dtype=str)
#test_imgs_path = np.loadtxt("../images/new_test_folder",dtype=str)

# Iterate over the test images
for images in test_imgs_path:
    img = Image.open(images).convert("RGB")
    print(img.size,img.mode)
    # Apply the transform to the input image
    img_tensor = transform(img)
    # Add a batch dimension to the tensor and pass it through the model
    output = model(img_tensor.unsqueeze(0))
    # Get the predicted class
    predicted_class = torch.argmax(output)
    print(f'The predicted class for Resnet is {predicted_class}')



