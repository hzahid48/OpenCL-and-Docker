#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    // Load kernel source code
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("rgb_to_gray.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Initialize OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Load colored image
    FILE *img_file;
    unsigned char *input_img;
    int width, height, channels;

    img_file = fopen("colored_image.jpg", "rb");
    if (!img_file) {
        fprintf(stderr, "Failed to open image file.\n");
        exit(1);
    }
    fseek(img_file, 0, SEEK_END);
    long file_size = ftell(img_file);
    fseek(img_file, 0, SEEK_SET);
    input_img = (unsigned char *)malloc(file_size);
    fread(input_img, 1, file_size, img_file);
    fclose(img_file);

    // Create OpenCL buffers
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, file_size, input_img, &ret);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, file_size, NULL, &ret);

    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "rgb_to_gray", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buf);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_buf);

    // Execute kernel
    size_t global_item_size[2] = {width, height};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

    // Read output buffer
    unsigned char *output_img = (unsigned char *)malloc(file_size);
    ret = clEnqueueReadBuffer(command_queue, output_buf, CL_TRUE, 0, file_size, output_img, 0, NULL, NULL);

    // Save grayscale image
    FILE *gray_img_file;
    gray_img_file = fopen("grayscale_image.jpg", "wb");
    fwrite(output_img, 1, file_size, gray_img_file);
    fclose(gray_img_file);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(input_buf);
    ret = clReleaseMemObject(output_buf);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(input_img);
    free(output_img);
    free(source_str);

    return 0;
}
