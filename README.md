# Generating Human Faces using DCGAN and SRGAN

This project uses Deep Convolutional Generative Adversarial Networks (DCGAN) and Super Resolution Generative Adversarial Networks (SRGAN) to generate realistic human faces. The models are trained on a dataset of celebrity faces and can be used to generate new, never-before-seen faces.

## Dataset

The dataset used in this project DCGAN part is the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which consists of over 200,000 celebrity images. The images are preprocessed and resized to 64x64 pixels for use in training the DCGAN model.

The dataset used in this project SRGAN part as ground truth image is the [DIV2K dataset]https://data.vision.ee.ethz.ch/cvl/DIV2K/). It consists of 800 high-quality images with a resolution of 2K (i.e., 2048x1080 pixels), which are divided into a training set of 800 images and a validation set of 100 images, which is used in Discriminator in SRGAN.

## Models

Two different models are used in this project: DCGAN and SRGAN.

### DCGAN

DCGAN is a type of generative adversarial network (GAN) that uses convolutional layers to generate images. In this project, the DCGAN model is trained on the CelebA dataset to generate realistic human faces. The generator and discriminator are both convolutional neural networks, and the model is trained using the binary cross-entropy loss function.

### SRGAN

SRGAN is a type of GAN that is used for super-resolution tasks, such as increasing the resolution of low-resolution images. In this project, the SRGAN model is used to generate high-resolution human faces from low-resolution inputs. The generator and discriminator are both convolutional neural networks, and the model is trained using the mean squared error loss function.

## Results

![Generated Face](generated face.png)

The DCGAN model is able to generate realistic human faces that closely resemble the faces in the CelebA dataset. The SRGAN model is able to generate high-resolution images from low-resolution inputs, resulting in much sharper and more detailed faces.

## Usage

To use the models to generate human faces, you can run the `generatingfaces_dcgan.ipynb` in Colab or Jupyter Notebook. This notebook takes as input the number of faces you want to generate. The generated images will be saved in the `generated_images` directory. Please refer to the notebook to see the instructions.

To use the models human faces generated for image super resolution, you can run the `SRGAN.ipynb` in Colab or Jupyter Notebook. This notebook first train SRGAN, then save weights of model in 'weights' directory. It takes as input directory of faces you want to generate and the type of model you want to use in SRGAN. The generated images will be saved in the directory you want to save the generated faces after SRGAN. Please refer to the notebook to see the instructions.

## Requirements

The following packages are required to run the code:

* TensorFlow 2.0 or higher
* NumPy
* Matplotlib
* OpenCV

## References

1. Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

2. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 105-114).

3. https://arxiv.org/abs/1511.06434v2
4. https://github.com/eriklindernoren/PyTorch-GAN
5. https://github.com/krasserm/super-resolution
6. https://arxiv.org/abs/1609.04802
7. https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
8. https://arxiv.org/pdf/2002.09797.pdf
9. https://github.com/ZSoumia/DCGAN-for-generating-human-faces-

