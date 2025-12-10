import onnx
from onnx import numpy_helper
from pathlib import Path

class ActivationFunc():
    @staticmethod
    def identity():
        return '#define ACTIVE(v) (v)'
    @staticmethod
    def relu():
        return '#define ACTIVE(v) (max(v, 0.0f))'
    @staticmethod
    def lrelu(n = 0.2):
        return f'#define ACTIVE(v) (max(v, v * {n:.7f}f))'

class ConvFunc():
    conv3x3_1to4_body = '''    const vec4 r0 = vec4(
        TEX_BIND_L0_TEXOFF(vec2(-1.0f,-1.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2( 0.0f,-1.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2( 1.0f,-1.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2(-1.0f, 0.0f)).x
    );
    const vec4 r4 = vec4(
        TEX_BIND_L0_TEXOFF(vec2( 0.0f, 0.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2( 1.0f, 0.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2(-1.0f, 1.0f)).x,
        TEX_BIND_L0_TEXOFF(vec2( 0.0f, 1.0f)).x
    );
    const float r8 = TEX_BIND_L0_TEXOFF(vec2( 1.0f, 1.0f)).x;

    float sum[4] = biases;
    for(int n = 0; n < 4; n++) {
        const vec4 k0 = vec4(kernels[n * 9 + 0], kernels[n * 9 + 1], kernels[n * 9 + 2], kernels[n * 9 + 3]);
        const vec4 k4 = vec4(kernels[n * 9 + 4], kernels[n * 9 + 5], kernels[n * 9 + 6], kernels[n * 9 + 7]);
        const float k8 = kernels[n * 9 + 8];
        sum[n] += (dot(r0, k0) + dot(r4, k4) + r8 * k8);
    }

    '''

    conv3x3_8to4_body = '''    const vec4 r[9][2] = vec4[9][2](
        vec4[2](TEX_BIND_L0_TEXOFF(vec2(-1.0f,-1.0f)), TEX_BIND_L1_TEXOFF(vec2(-1.0f,-1.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 0.0f,-1.0f)), TEX_BIND_L1_TEXOFF(vec2( 0.0f,-1.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 1.0f,-1.0f)), TEX_BIND_L1_TEXOFF(vec2( 1.0f,-1.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2(-1.0f, 0.0f)), TEX_BIND_L1_TEXOFF(vec2(-1.0f, 0.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 0.0f, 0.0f)), TEX_BIND_L1_TEXOFF(vec2( 0.0f, 0.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 1.0f, 0.0f)), TEX_BIND_L1_TEXOFF(vec2( 1.0f, 0.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2(-1.0f, 1.0f)), TEX_BIND_L1_TEXOFF(vec2(-1.0f, 1.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 0.0f, 1.0f)), TEX_BIND_L1_TEXOFF(vec2( 0.0f, 1.0f))),
        vec4[2](TEX_BIND_L0_TEXOFF(vec2( 1.0f, 1.0f)), TEX_BIND_L1_TEXOFF(vec2( 1.0f, 1.0f)))
    );

    float sum[4] = biases;
    for (int n = 0; n < 4; n++) {
        for (int i = 0; i < 9; i++) {
            const vec4 k0 = vec4(kernels[n * 8 * 9 + 8 * i + 0], kernels[n * 8 * 9 + 8 * i + 1], kernels[n * 8 * 9 + 8 * i + 2], kernels[n * 8 * 9 + 8 * i + 3]);
            const vec4 k1 = vec4(kernels[n * 8 * 9 + 8 * i + 4], kernels[n * 8 * 9 + 8 * i + 5], kernels[n * 8 * 9 + 8 * i + 6], kernels[n * 8 * 9 + 8 * i + 7]);
            sum[n] += (dot(k0, r[i][0]) + dot(k1, r[i][1]));
        }
    }

    '''

    deconv2x2_8to1_body = '''    vec2 fcoord = fract(TEX_BIND_L0_POS * TEX_BIND_L0_SIZE);
    vec2 pos = TEX_BIND_L0_POS + (vec2(0.5f) - fcoord) * TEX_BIND_L0_PT;

    ivec2 icoord = ivec2(fcoord * vec2(2.0f));
    int index = icoord.y * 2 + icoord.x;

    vec4 r0 = TEX_BIND_L0_TEX(pos);
    vec4 r1 = TEX_BIND_L1_TEX(pos);

    vec4 k0 = vec4(kernels[8 * index + 0], kernels[8 * index + 1], kernels[8 * index + 2], kernels[8 * index + 3]);
    vec4 k1 = vec4(kernels[8 * index + 4], kernels[8 * index + 5], kernels[8 * index + 6], kernels[8 * index + 7]);

    return vec4(clamp(dot(r0, k0) + dot(r1, k1), 0.0f, 1.0f), 0.0f, 0.0f, 1.0f);'''

    pixelshuffle_4to1_body = '''    vec2 fcoord = fract(TEX_BIND_L0_POS * TEX_BIND_L0_SIZE);
    vec2 pos = TEX_BIND_L0_POS + (vec2(0.5f) - fcoord) * TEX_BIND_L0_PT;

    ivec2 icoord = ivec2(fcoord * vec2(2.0f));
    int index = icoord.y * 2 + icoord.x;

    vec4 r = TEX_BIND_L0_TEX(pos);

    return vec4(clamp(float[4](r.x, r.y, r.z, r.w)[index], 0.0f, 1.0f), 0.0f, 0.0f, 1.0f);'''
    @staticmethod
    def conv3x3_1to4():
        return ConvFunc.conv3x3_1to4_body + f'return ACTIVE(vec4(sum[0], sum[1], sum[2], sum[3]));'
    @staticmethod
    def conv3x3_8to4():
        return ConvFunc.conv3x3_8to4_body + f'return ACTIVE(vec4(sum[0], sum[1], sum[2], sum[3]));'

    @staticmethod
    def conv3x3_8to4_residual(scale):
        return ConvFunc.conv3x3_8to4_body + f'return ACTIVE(vec4(sum[0], sum[1], sum[2], sum[3]) * {scale:.7f}f + TEX_BIND_L2_TEXOFF(vec2(0.0f, 0.0f)));'
    @staticmethod
    def conv3x3_8to4_residual_add(scale):
        return ConvFunc.conv3x3_8to4_body + f'return ACTIVE(vec4(sum[0], sum[1], sum[2], sum[3]) * {scale:.7f}f + TEX_BIND_L2_TEXOFF(vec2(0.0f, 0.0f)) + TEX_BIND_L3_TEXOFF(vec2(0.0f, 0.0f)));'
    @staticmethod
    def deconv2x2_8to1():
        return ConvFunc.deconv2x2_8to1_body
    @staticmethod
    def pixelshuffle_4to1():
        return ConvFunc.pixelshuffle_4to1_body

class HookBlock():
    def __init__(self):
        super(HookBlock, self).__init__()
        self.control = ''
        self.defines = []
        self.const_datas = []
        self.funcions = []
        self.bind_textures = []
        self.save_texture = ''

    def hook(self, channel):
        self.control += f'//!HOOK {channel}\n'
        return self

    def when(self, scale_factor):
        self.control += f'//!WHEN OUTPUT.w LUMA.w / {scale_factor} > OUTPUT.h LUMA.h / {scale_factor} > *\n'
        return self

    def desc(self, info):
        self.control += f'//!DESC {info}\n'
        return self

    def bind(self, texture):
        self.control += f'//!BIND {texture}\n'
        self.bind_textures.append(texture)
        return self

    def save(self, texture):
        self.control += f'//!SAVE {texture}\n'
        self.save_texture = texture
        return self

    def height(self, h):
        self.control += f'//!HEIGHT LUMA.h {h} *\n'
        return self

    def width(self, w):
        self.control += f'//!WIDTH LUMA.w {w} *\n'
        return self

    def components(self, num):
        self.control += f'//!COMPONENTS {num}\n'
        return self

    def define_activation_func(self, func):
        self.defines.append(f'{func}\n')
        return self

    def define_const_array_data(self, name, size, data):
        array_data = [f'const float {name}[{size}] = float[{size}](']
        for i in range(0, len(data), 8):
            line = ', '.join(f'{x:+.7f}f' for x in data[i:i+8])
            if i + 8 >= len(data):
                array_data.append(f'  {line}')
            else:
                array_data.append(f'  {line},')
        array_data.append(');\n')
        self.const_datas.append('\n'.join(array_data))
        return self

    def define_hook_func(self, func):
        hook_func = [f'vec4 hook() {{']
        hook_func.append(f'{func}')
        hook_func.append('}\n')
        self.funcions.append('\n'.join(hook_func))
        return self

    def get_glsl(self):
        count = 0
        for tex in self.bind_textures:
            self.defines.append(f'#define TEX_BIND_L{count}_TEXOFF {tex}_texOff')
            self.defines.append(f'#define TEX_BIND_L{count}_TEX {tex}_tex')
            self.defines.append(f'#define TEX_BIND_L{count}_POS {tex}_pos')
            self.defines.append(f'#define TEX_BIND_L{count}_SIZE {tex}_size')
            self.defines.append(f'#define TEX_BIND_L{count}_PT {tex}_pt\n')
            count += 1
        return '\n'.join([self.control, *self.defines, *self.const_datas, *self.funcions])

def get_onnx_weights_grouped(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = {
            'array': numpy_helper.to_array(init),
            'name': init.name
        }

    kernels_list = []
    biases_list = []

    for node in graph.node:
        if node.op_type in ['Conv', 'ConvTranspose']:
            weight_array = None
            bias_array = None

            if len(node.input) >= 2:
                weight_name = node.input[1]
                if weight_name in initializers:
                    weight_array = initializers[weight_name]['array']

            if len(node.input) >= 3:
                bias_name = node.input[2]
                if bias_name in initializers:
                    bias_array = initializers[bias_name]['array']

            if weight_array is None:
                continue

            if node.op_type == 'Conv':
                weight_nhwc = weight_array.transpose(0, 2, 3, 1)
            else:  # ConvTranspose
                weight = weight_array.transpose(1, 0, 2, 3)
                weight_nhwc = weight.transpose(0, 2, 3, 1)

            N, H, W, C = weight_nhwc.shape

            layer_kernels = []
            for n_start in range(0, N, 4):
                n_end = min(n_start + 4, N)
                group_weight = weight_nhwc[n_start:n_end]
                layer_kernels.append(group_weight.flatten().tolist())

            kernels_list.append(layer_kernels)

            if bias_array is not None:
                layer_biases = []
                for n_start in range(0, N, 4):
                    n_end = min(n_start + 4, N)
                    group_bias = bias_array[n_start:n_end]
                    layer_biases.append(group_bias.tolist())
                biases_list.append(layer_biases)
            else:
                biases_list.append([])

    return kernels_list, biases_list

def acnet_glsl(output_file, onnx_file, limit_factor = 1.2):
    kernels_list, biases_list = get_onnx_weights_grouped(onnx_file)

    l = 0
    blocks = []
    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ACNet Layer{l} L0')
            .when(limit_factor)
            .bind('LUMA')
            .save('TMP1_L0')
            .components(4)
            .define_activation_func(ActivationFunc.relu())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_1to4())
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ACNet Layer{l} L1')
            .when(limit_factor)
            .bind('LUMA')
            .save('TMP1_L1')
            .components(4)
            .define_activation_func(ActivationFunc.relu())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_1to4())
            .get_glsl()
    )
    l += 1

    for _ in range(4):
        kernels = kernels_list[l][0]
        biases =  biases_list[l][0]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ACNet Layer{l} L0')
                .when(limit_factor)
                .bind('TMP1_L0')
                .bind('TMP1_L1')
                .save('TMP2_L0')
                .components(4)
                .define_activation_func(ActivationFunc.relu())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        kernels = kernels_list[l][1]
        biases =  biases_list[l][1]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ACNet Layer{l} L1')
                .when(limit_factor)
                .bind('TMP1_L0')
                .bind('TMP1_L1')
                .save('TMP2_L1')
                .components(4)
                .define_activation_func(ActivationFunc.relu())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        l += 1

        kernels = kernels_list[l][0]
        biases =  biases_list[l][0]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ACNet Layer{l} L0')
                .when(limit_factor)
                .bind('TMP2_L0')
                .bind('TMP2_L1')
                .save('TMP1_L0')
                .components(4)
                .define_activation_func(ActivationFunc.relu())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        kernels = kernels_list[l][1]
        biases =  biases_list[l][1]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ACNet Layer{l} L1')
                .when(limit_factor)
                .bind('TMP2_L0')
                .bind('TMP2_L1')
                .save('TMP1_L1')
                .components(4)
                .define_activation_func(ActivationFunc.relu())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        l += 1

    kernels = kernels_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ACNet Layer{l} L0')
            .width(2)
            .height(2)
            .bind('TMP1_L0')
            .bind('TMP1_L1')
            .components(1)
            .define_activation_func(ActivationFunc.relu())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_hook_func(ConvFunc.deconv2x2_8to1())
            .get_glsl()
    )

    glsl = '\n'.join(blocks)

    with open(output_file, "w") as f:
        f.write(glsl)

def arnet_glsl(output_file, onnx_file, arnet_blocks, limit_factor = 1.2):
    kernels_list, biases_list = get_onnx_weights_grouped(onnx_file)

    l = 0
    blocks = []
    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('LUMA')
            .save('FEAT_L0')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_1to4())
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
            .when(limit_factor)
            .bind('LUMA')
            .save('FEAT_L1')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_1to4())
            .get_glsl()
    )
    l += 1

    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('FEAT_L0')
            .bind('FEAT_L1')
            .save('TMP1_L0')
            .components(4)
            .define_activation_func(ActivationFunc.lrelu(0.2))
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4())
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
            .when(limit_factor)
            .bind('FEAT_L0')
            .bind('FEAT_L1')
            .save('TMP1_L1')
            .components(4)
            .define_activation_func(ActivationFunc.lrelu(0.2))
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4())
            .get_glsl()
    )
    l += 1

    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('TMP1_L0')
            .bind('TMP1_L1')
            .bind('FEAT_L0')
            .save('TMP2_L0')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4_residual(0.2))
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
            .when(limit_factor)
            .bind('TMP1_L0')
            .bind('TMP1_L1')
            .bind('FEAT_L1')
            .save('TMP2_L1')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4_residual(0.2))
            .get_glsl()
    )
    l += 1

    for _ in range(arnet_blocks - 2):
        kernels = kernels_list[l][0]
        biases =  biases_list[l][0]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
                .when(limit_factor)
                .bind('TMP2_L0')
                .bind('TMP2_L1')
                .save('TMP1_L0')
                .components(4)
                .define_activation_func(ActivationFunc.lrelu(0.2))
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        kernels = kernels_list[l][1]
        biases =  biases_list[l][1]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
                .when(limit_factor)
                .bind('TMP2_L0')
                .bind('TMP2_L1')
                .save('TMP1_L1')
                .components(4)
                .define_activation_func(ActivationFunc.lrelu(0.2))
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4())
                .get_glsl()
        )
        l += 1

        kernels = kernels_list[l][0]
        biases =  biases_list[l][0]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
                .when(limit_factor)
                .bind('TMP1_L0')
                .bind('TMP1_L1')
                .bind('TMP2_L0')
                .save('TMP2_L0')
                .components(4)
                .define_activation_func(ActivationFunc.identity())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4_residual(0.2))
                .get_glsl()
        )
        kernels = kernels_list[l][1]
        biases =  biases_list[l][1]
        blocks.append(
            HookBlock()
                .hook('LUMA')
                .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
                .when(limit_factor)
                .bind('TMP1_L0')
                .bind('TMP1_L1')
                .bind('TMP2_L1')
                .save('TMP2_L1')
                .components(4)
                .define_activation_func(ActivationFunc.identity())
                .define_const_array_data("kernels", len(kernels), kernels)
                .define_const_array_data("biases", len(biases), biases)
                .define_hook_func(ConvFunc.conv3x3_8to4_residual(0.2))
                .get_glsl()
        )
        l += 1

    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('TMP2_L0')
            .bind('TMP2_L1')
            .save('TMP1_L0')
            .components(4)
            .define_activation_func(ActivationFunc.lrelu(0.2))
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4())
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
            .when(limit_factor)
            .bind('TMP2_L0')
            .bind('TMP2_L1')
            .save('TMP1_L1')
            .components(4)
            .define_activation_func(ActivationFunc.lrelu(0.2))
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4())
            .get_glsl()
    )
    l += 1

    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('TMP1_L0')
            .bind('TMP1_L1')
            .bind('TMP2_L0')
            .bind('FEAT_L0')
            .save('TMP2_L0')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4_residual_add(0.2))
            .get_glsl()
    )
    kernels = kernels_list[l][1]
    biases =  biases_list[l][1]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L1')
            .when(limit_factor)
            .bind('TMP1_L0')
            .bind('TMP1_L1')
            .bind('TMP2_L1')
            .bind('FEAT_L1')
            .save('TMP2_L1')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4_residual_add(0.2))
            .get_glsl()
    )
    l += 1

    kernels = kernels_list[l][0]
    biases =  biases_list[l][0]
    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ARNet b{arnet_blocks} Layer{l} L0')
            .when(limit_factor)
            .bind('TMP2_L0')
            .bind('TMP2_L1')
            .save('TMP1_L0')
            .components(4)
            .define_activation_func(ActivationFunc.identity())
            .define_const_array_data("kernels", len(kernels), kernels)
            .define_const_array_data("biases", len(biases), biases)
            .define_hook_func(ConvFunc.conv3x3_8to4())
            .get_glsl()
    )
    l += 1

    blocks.append(
        HookBlock()
            .hook('LUMA')
            .desc(f'ACNet b{arnet_blocks} Layer{l} L0')
            .width(2)
            .height(2)
            .bind('TMP1_L0')
            .components(1)
            .define_hook_func(ConvFunc.pixelshuffle_4to1())
            .get_glsl()
    )

    glsl = '\n'.join(blocks)

    with open(output_file, "w") as f:
        f.write(glsl)

if __name__ == '__main__':
    Path('glsl').mkdir(exist_ok=True)

    acnet_glsl('glsl/acnet_gan.glsl', 'models/acnet_gan.onnx', 1.2)
    acnet_glsl('glsl/acnet_hdn0.glsl', 'models/acnet_hdn0.onnx', 1.2)
    acnet_glsl('glsl/acnet_hdn1.glsl', 'models/acnet_hdn1.onnx', 1.2)
    acnet_glsl('glsl/acnet_hdn2.glsl', 'models/acnet_hdn2.onnx', 1.2)
    acnet_glsl('glsl/acnet_hdn3.glsl', 'models/acnet_hdn3.onnx', 1.2)

    arnet_glsl('glsl/arnet_b4_le.glsl', 'models/arnet_b4_le.onnx', 4, 1.2)
    arnet_glsl('glsl/arnet_b4_hdn.glsl', 'models/arnet_b4_hdn.onnx', 4, 1.2)

    arnet_glsl('glsl/arnet_b8_le.glsl', 'models/arnet_b8_le.onnx', 8, 1.2)
    arnet_glsl('glsl/arnet_b8_hdn.glsl', 'models/arnet_b8_hdn.onnx', 8, 1.2)

    arnet_glsl('glsl/arnet_b16_le.glsl', 'models/arnet_b16_le.onnx', 16, 1.2)
    arnet_glsl('glsl/arnet_b16_hdn.glsl', 'models/arnet_b16_hdn.onnx', 16, 1.2)

    arnet_glsl('glsl/arnet_b24_le.glsl', 'models/arnet_b24_le.onnx', 24, 1.2)
    arnet_glsl('glsl/arnet_b24_hdn.glsl', 'models/arnet_b24_hdn.onnx', 24, 1.2)

    arnet_glsl('glsl/arnet_b32_le.glsl', 'models/arnet_b32_le.onnx', 32, 1.2)
    arnet_glsl('glsl/arnet_b32_hdn.glsl', 'models/arnet_b32_hdn.onnx', 32, 1.2)

    arnet_glsl('glsl/arnet_b48_le.glsl', 'models/arnet_b48_le.onnx', 48, 1.2)
    arnet_glsl('glsl/arnet_b48_hdn.glsl', 'models/arnet_b48_hdn.onnx', 48, 1.2)

    arnet_glsl('glsl/arnet_b64_le.glsl', 'models/arnet_b64_le.onnx', 64, 1.2)
    arnet_glsl('glsl/arnet_b64_hdn.glsl', 'models/arnet_b64_hdn.onnx', 64, 1.2)
