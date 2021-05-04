# ACNetGLSL
ACNet is a CNN algorithm, implemented by [Anime4KCPP](https://github.com/TianZerL/Anime4KCPP), it aims to provide both high-quality and high-performance.  
This GLSL implementation can be used in [MPV player](https://mpv.io), it is cross-platform. Windows users can also use [Anime4KCPP DirectShow Filter](https://github.com/TianZerL/Anime4KCPP/releases) for MPC-HC/BE or potplayer.
# How to use
1. Download the glsl file and MPV player.
2. copy glsl to the root directory of MPV. 
3. create a file named `mpv.conf` in the root directory of MPV, and add the following statement (Assume the glsl file name is ACNet.glsl): 

    ```conf
    profile=gpu-hq
    glsl-shader="~~/ACNet.glsl"
    ```
4. When playing the video, press `Shift + i` and then `2` to check if it is enabled.
5. You may also switch the shader by keyboard if you can edit input.conf.
    ```input.conf
    Meta+0 no-osd change-list glsl-shaders clr ""; show-text "GLSL shaders cleared"
    Meta+1 no-osd change-list glsl-shaders set "~/.config/mpv/shaders/ACNet_HDN_L1.glsl"; show-text "Anime4k: ACNet_HDN_L1"
    Meta+2 no-osd change-list glsl-shaders set "~/.config/mpv/shaders/ACNet_HDN_L2.glsl"; show-text "Anime4k: ACNet_HDN_L2"
    Meta+3 no-osd change-list glsl-shaders set "~/.config/mpv/shaders/ACNet_HDN_L3.glsl"; show-text "Anime4k: ACNet_HDN_L3"
    ```

