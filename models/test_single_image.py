import torch
import torchvision.transforms as transforms
from PIL import Image
# Define the transform to apply to the input image
# defining the transforms and dataloaders
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create an instance of the model
#model = torch.load("../new_saved_models/newLenet.pt", map_location=torch.device('cpu'))
model = torch.load("../saved_models/newResnet.pt", map_location=torch.device('cpu'))
model.eval()
# Load the input image
#img = Image.open("../images/new_image_samples/35767.jpg").convert("RGB")
img = Image.open("../images/new_test_samples/neutral1.jpg").convert("RGB")
#img = Image.open("../images/new_google_test_images/happy_image.jpeg").convert("RGB")
print(img.size,img.mode)
# Apply the transform to the input image
img_tensor = transform(img)
# Add a batch dimension to the tensor and pass it through the model
output = model(img_tensor.unsqueeze(0))
# Get the predicted class
predicted_class = torch.argmax(output)
print(f'The predicted class for neutral, Resnet is {predicted_class}')
