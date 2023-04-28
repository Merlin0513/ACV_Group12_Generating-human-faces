# Generating Human Faces using DCGAN and SRGAN

This project uses Deep Convolutional Generative Adversarial Networks (DCGAN) and Super Resolution Generative Adversarial Networks (SRGAN) to generate realistic human faces. The models are trained on a dataset of celebrity faces and can be used to generate new, never-before-seen faces.

## Dataset

The dataset used in this project is the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which consists of over 200,000 celebrity images. The images are preprocessed and resized to 64x64 pixels for use in training the DCGAN model.

## Models

Two different models are used in this project: DCGAN and SRGAN.

### DCGAN

DCGAN is a type of generative adversarial network (GAN) that uses convolutional layers to generate images. In this project, the DCGAN model is trained on the CelebA dataset to generate realistic human faces. The generator and discriminator are both convolutional neural networks, and the model is trained using the binary cross-entropy loss function.

### SRGAN

SRGAN is a type of GAN that is used for super-resolution tasks, such as increasing the resolution of low-resolution images. In this project, the SRGAN model is used to generate high-resolution human faces from low-resolution inputs. The generator and discriminator are both convolutional neural networks, and the model is trained using the mean squared error loss function.

## Results

![Generated Face](Results images/face.png)
The DCGAN model is able to generate realistic human faces that closely resemble the faces in the CelebA dataset. The SRGAN model is able to generate high-resolution images from low-resolution inputs, resulting in much sharper and more detailed faces.

## Usage

To use the models to generate human faces, you can run the `generate_faces.py` script. This script takes as input the number of faces you want to generate and the type of model you want to use (DCGAN or SRGAN). The generated images will be saved in the `generated_images` directory.

## Requirements

The following packages are required to run the code:

* TensorFlow 2.0 or higher
* NumPy
* Matplotlib
* OpenCV

## References

1. Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

2. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 105-114).
