import torch
import timm
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True,
    img_size=224
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():

        features = model.forward_features(img_tensor)

        embedding = features[:, 0]

    return embedding.squeeze().cpu().numpy()


