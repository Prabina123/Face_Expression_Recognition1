import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_imgs_path = np.loadtxt("../test_images.txt", dtype=str)
test_labels = np.loadtxt("../test_labels.txt", dtype=np.int64)

num_correct = 0
num_total = 0

# Create an instance of the model
model = torch.load("../new_saved_models/newResnet.pt", map_location=torch.device('cpu'))
#model = torch.load("../new_saved_models/newResnet.pt", map_location=torch.device('cpu'))
model.eval()

# Iterate over the test images
for i, img_path in enumerate(test_imgs_path):
    #img = Image.open(img_path).convert("RGB")
    img = Image.open("../images/new_test_samples/surprise1.jpg").convert("RGB")
    # Apply the transform to the input image
    img_tensor = transform(img)
    # Add a batch dimension to the tensor and pass it through the model
    output = model(img_tensor.unsqueeze(0))
    # Get the predicted class
    predicted_class = torch.argmax(output)
    # Convert the predicted class to a string label
    if predicted_class == 0:
        pred_label = "angry"
    elif predicted_class == 1:
        pred_label = "disguist"
    elif predicted_class == 2:
        pred_label = "fear"
    elif predicted_class == 3:
        pred_label = "happy"
    elif predicted_class == 4:
        pred_label = "neutral"
    elif predicted_class == 5:
        pred_label = "sad"
    elif predicted_class == 6:
        pred_label = "surprise"

    # Compare the predicted class with the true label and update the accuracy
    if predicted_class == test_labels[i]:
        num_correct += 1
    num_total += 1
        

    print(f"Image {i+1}: Predicted label: {pred_label}, True label: {test_labels[i]}")

test_accuracy = num_correct / num_total
print(f"Test accuracy: {test_accuracy:.4f}")
