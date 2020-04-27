# MICCAI 2018 Challenge-Nuclei Segmentation
This was a challenge held as part of the MICCAI 2018 conference. The task is to segment to the nuclei present in a given image of a tissue.
My solution uses a deep learning approach for the task of semantic segmentation.

##Data
The challenge provided image data and their respective ground truth masks of uneven dimensions. For the sake of uniform dimensions for training, we created overlapped patches of the images and masks, of size (256,256), using create_patch.py.

![Input Image][\images\image.png]![Ground Truth mask][\images\mask.png]
