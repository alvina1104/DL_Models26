from fastapi import UploadFile, File, HTTPException, APIRouter
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image


check_image_router = APIRouter(prefix='/predict', tags=['MNIST'])


class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
  )

    self.second = nn.Sequential(
      nn.Flatten(),
      nn.Linear(16 * 14 * 14, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
  )

  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('mysite/dl_models/mnist_model.pth', map_location=device))
model.to(device)
model.eval()


@check_image_router.post('/')
async def check_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Image is empty')

        img = Image.open(io.BytesIO(image_bytes))
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return { 'Answer': pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

