from fastapi import HTTPException, UploadFile, File, APIRouter
import uvicorn
import torch
import torch.nn as nn
import io
from torchvision import transforms
from PIL import Image


scene_router = APIRouter(prefix='/scene_predict', tags=['Image Scene'])


class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),   # 🔥 көбөйттүк
            nn.Linear(512, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transform_data = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('mysite/dl_models/scene_model2.pth', map_location=device))
model.to(device)
model.eval()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

@scene_router.post('/')
async def scene_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=400, detail='No such File')

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = transform_data(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)

            class_id = torch.argmax(probs, dim=1).item()
            label = class_names[class_id]

        return {
            "class_id": class_id,
            "label": label,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


