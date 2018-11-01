
CycleGAN in MxNetR
===

This is a simple example for implementing the CycleGAN by MxNetR. The idea is devoloped by **[this paper](https://arxiv.org/abs/1703.10593)**, and related details can be found in this [project](https://junyanz.github.io/CycleGAN/). This is a simple example for MxNetR user, I will use a relatively small dataset for demonstrating how it work. 

Note: this is an example for implementing CycleGAN generated Monet paintings from photos. The dataset is released by [junyanz/CycleGAN](https://github.com/junyanz/CycleGAN), and you can download it from [https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets).

# The principle of CycleGAN

The CycleGAN architecture is showed as following. The nice explanation by Hardik Bansal and Archit Rathore, with Tensorflow code documentation ([Understanding and Implementing CycleGAN](https://hardikbansal.github.io/CycleGANBlog/)). The only thing the blog doesn’t mention is 'identity mapping loss'. So the revised architecture need to include this loss.

<p align="center">
  <img src="images/CycleGAN.gif">
</p>

The identity mapping loss helps preserve the color of the input paintings. For example, this photo is the input as following:

<p align="center">
  <img src="images/input_photo.png">
</p>

If the CycleGAN does not include identity mapping loss, after 50 epoch, even though he has performed quite well, but it will cause conversion distortion.

<p align="center">
  <img src="images/fake_monet (v1).png">
</p>

If we revise the CycleGAN and include identity mapping loss, after 50 epoch, it will perform well.

<p align="center">
  <img src="images/fake_monet (v2-1).png">
</p>

More interestingly, we can also process this image again through the photo2monet model as following.

<p align="center">
  <img src="images/fake_monet (v2-2).png">
</p>

One more!

<p align="center">
  <img src="images/fake_monet (v2-3).png">
</p>

CycleGAN is a very cool model, let's try to implement it!

# If you just want to use this model.

You can use the code ["1. Photo2Monet.R"](https://github.com/xup6fup/MxNetR-CycleGAN/blob/master/code/3.%20Use%20model/1.%20Photo2Monet.R) for processing a photo. Here we prepared a well-trained model for your experiment. The 'P2M_gen_v1-0000.params' and 'P2M_gen_v1-symbol.json' can be found in the folder 'well trained model'. This is a CycleGAN trained without identity mapping loss. The version 2 model is to use the identity mapping loss for training. You can find 'P2M_gen_v2-0000.params' and 'P2M_gen_v2-symbol.json'. Here we use the 'input_photo.png' for testing the model.

# If you want to train a CycleGAN.


