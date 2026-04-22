from fastapi import FastAPI
import uvicorn
from mysite.api import mnist, fashion, cifar, flowers, cifar100, scene, fruits, trash

deep_app = FastAPI()
deep_app.include_router(mnist.check_image_router)
deep_app.include_router(fashion.fashion_router)
deep_app.include_router(cifar.cifar_router)
deep_app.include_router(flowers.flowers_router)
deep_app.include_router(cifar100.cifar100_router)
deep_app.include_router(scene.scene_router)
deep_app.include_router(fruits.fruits_router)
deep_app.include_router(trash.trash_router)


if __name__ == "__main__":
    uvicorn.run(deep_app, host="127.0.0.1", port=8000)


