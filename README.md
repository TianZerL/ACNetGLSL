# ACNetGLSL
This repository contains the GLSL implementation of the deep learning models used by the [Anime4KCPP](https://github.com/TianZerL/Anime4KCPP) project for the [MPV player](https://mpv.io). These models are designed to deliver high performance and quality. For Windows users, you can also use the [Anime4KCPP DirectShow Filter](https://github.com/TianZerL/Anime4KCPP/releases), which is tailored for DirectShow-based players such as MPC-BE, MPC-HC, and PotPlayer.

# Models
The ACNet and ARNet models have been redesigned and retrained. The previous ACNet model is now referred to as ACNet-Legacy.

The ACNet-Legacy model includes 5 variants, from hdn0 to hdn3, with progressively stronger denoising capabilities. Additionally, a gan variant has been introduced to enhance details.

ACNet and ARNet offer models of various sizes, which can be selected based on your hardware configuration. Each size provides 4 variants:

- no suffix: trained to be neutral.
- hdn: trained to mildly denoise.
- box: trained to be neutral through box degradation, which is better for line restoration, but may appear slightly blurry.
- box-hdn: trained to mildly denoise based on the box variant.

If you are interested in the training of ACNet and ARNet, or the scripts for exporting them to GLSL, please check out the [training code](https://github.com/TianZerL/RealTimeAnimeSISR).

# How to use
1. Download the GLSL files and place them in any location you prefer.
2. In `mpv.conf`, add the line: `glsl-shader="path/to/your/shader.glsl"`
3. For larger models, the initial compilation of the shaders may take a considerable amount of time, which could cause MPV to become unresponsive for a while.
4. While playing the video, press `Shift + i` followed by `2` to check if it is enabled. If successful, you should see items containing 'ACNet' or 'ARNet'.

Specifically, for Windows users, the simplest method is to create a folder named `portable_config` in the root directory of your MPV installation. Copy all the GLSL files into this folder. Then, create a new `mpv.conf` file and write the following content to enable `arnet_f8b8_hdn` model:
```conf
glsl-shader="~~/arnet_f8b8_hdn.glsl"
```
