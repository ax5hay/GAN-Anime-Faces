# GAN · Anime Faces

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Task](https://img.shields.io/badge/task-image_generation-4a4a52?style=flat-square)

A generative adversarial network that learns to **synthesise anime faces** from the
[Kaggle anime-faces dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces).
A generator turns random noise into 64×64 faces while a discriminator learns to tell real from
fake; the two train against each other until the generator's output is convincing.

## What's inside

| File | Role |
|------|------|
| `Akshay_Bajpai_GAN-anime-faces.ipynb` | Full notebook: data pipeline, generator + discriminator, training loop, sample grids |

> The dataset and generated images exceeded GitHub's upload limit, so only the notebook is
> committed. Point the data loader at the Kaggle dataset above to reproduce.

## Approach

1. **Data** — stream the face images, normalise to `[-1, 1]` for a `tanh` generator output.
2. **Generator** — dense projection → stacked `Conv2DTranspose` upsampling blocks → 64×64×3.
3. **Discriminator** — strided `Conv2D` downsampling → single real/fake logit.
4. **Adversarial loop** — alternate discriminator and generator updates with binary cross-entropy; snapshot a sample grid every few epochs to watch faces emerge.

## Run it

```bash
pip install tensorflow numpy matplotlib
# download the Kaggle anime-faces dataset and set its path in the notebook
jupyter notebook Akshay_Bajpai_GAN-anime-faces.ipynb
```

> Early deep-learning project — trained on CPU first, then moved to a GPU environment for later runs. See the notebook for the architecture and sample outputs.
