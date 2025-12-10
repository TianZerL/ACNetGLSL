# ACNetGLSL
This repository contains the GLSL implementation of the deep learning models used by the [Anime4KCPP](https://github.com/TianZerL/Anime4KCPP) project for the [MPV player](https://mpv.io). These models are designed to deliver high performance and quality. For Windows users, you can also use the [Anime4KCPP DirectShow Filter](https://github.com/TianZerL/Anime4KCPP/releases), which is tailored for DirectShow-based players such as MPC-BE, MPC-HC, and PotPlayer.

# Models
The ACNet model remains the same as before, but it has been uniformly updated with a new implementation, which is theoretically faster. The ACNet model includes 5 variants, from HDN0 to HDN3, with progressively stronger denoising capabilities. Additionally, a GAN variant has been introduced to enhance details.

The ARNet series includes 7 models of different sizes, ranging from B4 to B64, with increasing depth and parameter counts, resulting in improved quality. Each size offers 2 variants: the LE variant enhances lines but may alter the visual style. If you prefer not to have this effect, you can use the HDN variant. The HDN variant of ARNet is designed to preserve the original image’s look and feel, with only mild denoising (even lighter than ACNet HDN0).

# How to use
1. Download the GLSL files and place them in any location you prefer.
2. In `mpv.conf`, add the line: `glsl-shader="path/to/your/shader.glsl"`
3. For larger models, the initial compilation of the shaders may take a considerable amount of time, which could cause MPV to become unresponsive for a while.
4. While playing the video, press `Shift + i` followed by `2` to check if it is enabled. If successful, you should see items containing 'ACNet' or 'ARNet'.

Specifically, for Windows users, the simplest method is to create a folder named `portable_config` in the root directory of your MPV installation. Copy all the GLSL files into this folder. Then, create a new `mpv.conf` file and write the following content:
```conf
profile=gpu-hq
glsl-shader="~~/arnet_b16_hdn.glsl"
```
