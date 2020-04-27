# MICCAI 2018 Challenge-Nuclei Segmentation
This was a challenge held as part of the MICCAI 2018 conference. The task is to segment to the nuclei present in a given image of a tissue.
My solution uses a deep learning approach for the task of semantic segmentation.

## Data
The challenge provided image data and their respective ground truth masks of uneven dimensions. For the sake of uniform dimensions for training, we created overlapped patches of the images and masks, of size (256,256), using create_patch.py. Given below is an example of an input patch and it's respective ground truth patch.

![Input Image](/images/image.png)                  ![Ground Truth mask](/images/mask.png)

## Architecture
I used the [UNet architecture for the task of semantic segmentation](https://arxiv.org/pdf/1505.04597.pdf). I implemented this architecture using Keras.
![Architecture](/images/unet.png)
## Training
The model is trained with a batch size of 4 over 2422 images for training and 484 images for validation. Run main.py for training. A validation accuracy of 93% was achieved with a dice score of 0.82.


