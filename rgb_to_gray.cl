__kernel void rgb_to_gray(__global const uchar4 *input,
                          __global uchar *output,
                          const int width,
                          const int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < width && j < height) {
        int index = j * width + i;
        uchar4 pixel = input[index];
        output[index] = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}
