import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from mysite.frontend.mnist_front import check_mnist
from mysite.frontend.fashion_front import check_fashion
from mysite.frontend.cifar_front import check_cifar
from mysite.frontend.flowers_front import check_flowers
from mysite.frontend.cifar100_front import check_cifar100
from mysite.frontend.scene_front import check_scene
from mysite.frontend.fruit_front import check_fruits
from mysite.frontend.trash_front import check_trash


with st.sidebar:
    name = st.radio('DL Models:', ['Info', 'MNIST', 'FashionMNIST', 'CIFAR-10', 'CIFAR-100','Flowers',
                                   'Image Scene', 'Fruits', 'Trash Image'])


if name == 'Info':
    st.title('Welcome to Deep Learning Models')
    st.markdown("""
    * MNIST — Handwritten digit recognition (0–9)
    * FashionMNIST — Clothing classification (t-shirts, shoes, bags, etc.)
    * CIFAR-10 — Image classification (animals, transport and other objects)
    * **CIFAR-100** — Object classification of 100 different classes (insects, trees, fish, people, etc.)
    * Flowers — Identifying types of flowers from an image
    * **Image Scene** — Scene type recognition (buildings, forest, glacier, mountain, sea, street)
    * **Fruits** — Classification of various fruits  (apples, bananas, grapes , etc.)
    * **Trash Image — Waste classification (cardboard, glass, metal, paper, plastic, trash)
    """)

elif name == 'MNIST':
    check_mnist()

elif name == 'FashionMNIST':
    check_fashion()

elif name == 'CIFAR-10':
    check_cifar()

elif name == 'Flowers':
    check_flowers()

elif name == 'CIFAR-100':
    check_cifar100()

elif name == 'Image Scene':
    check_scene()

elif name == 'Fruits':
    check_fruits()

elif name == 'Trash Image':
    check_trash()