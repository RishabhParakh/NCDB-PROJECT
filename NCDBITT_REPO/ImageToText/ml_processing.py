import torch
from torchvision import transforms, models
from PIL import Image
from trainmodel import CarModelCNN, car_classes

# Define the paths and transforms
MODEL_PATH = '/home/ncdbproj/NCDBContent/ImageToText/car_model_recognition.pth'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CarModelCNN(num_classes=len(car_classes) + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Load a pre-trained ResNet model to identify if the image is a car
car_identifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
car_identifier.eval()

def is_car_image(image_path):
    """Check if the uploaded image is likely to be a car."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = car_identifier(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Identify the top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        car_related_classes = [407, 436, 511, 656, 817]  # Typical car-related classes in ImageNet

        # Check if one of the top 5 predictions is a car-related class
        for i in range(top5_catid.size(0)):
            if top5_catid[i].item() in car_related_classes:
                return True

    print("This is not a car")
    return False

def process_image(image_path):
    if not is_car_image(image_path):
        return "Sorry, this is not a car. Please upload a car image."

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = car_classes[predicted.item()] if predicted.item() < len(car_classes) else "I don't know this car model"

        # Calculate similarity scores with all labels in the dataset
        similarity_scores = torch.nn.functional.softmax(output, dim=1)
        top_score, top_class = torch.topk(similarity_scores, 1)

    if top_score[0].item() > 0.9:
        return f"it's a {predicted_label}"
    elif top_score[0].item() > 0.3:
        return f"I'm not sure, but it might be a {predicted_label}"
    else:
        return "yoo I don't know this car model"


