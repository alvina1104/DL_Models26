from fastapi import  UploadFile, File, HTTPException, APIRouter
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

fashion_router = APIRouter(prefix='/fashion_predict', tags=['Fashion MNIST'])


labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


class FashionImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FashionImage()
model.load_state_dict(torch.load('mysite/dl_models/fashion_model.pth', map_location=device))
model.to(device)
model.eval()


@fashion_router.post('/')
async def fashion_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Image is empty')

        img = Image.open(io.BytesIO(image_bytes))

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()


        predicted_label = labels[pred]

        return {
            "class_id": pred,
            "label": predicted_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

