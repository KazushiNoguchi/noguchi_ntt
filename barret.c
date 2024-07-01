#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_CL_ERROR(err, msg) if (err != CL_SUCCESS) { printf("%s failed: %d\n", msg, err); exit(1); }

const char *kernelsource = "__kernel void barrett_reduction( \n" \
"   const long p,                                 \n" \
"   const long m,                                 \n" \
"   __global long *input,                         \n" \
"   __global long *output) {                      \n" \
"       int id = get_global_id(0);                \n" \
"       long d = input[id];                       \n" \
"       long q = ((d * m) >> 52);                 \n" \
"       d = d - q * p;                            \n" \
"       if (d >= p) d -= p;                       \n" \
"       if (d < 0) d += p;                        \n" \
"       output[id] = d;                           \n" \
"}                                                \n";

void cpu_barrett_reduction(long p, long m, long *input, long *output, long N) {
    for (long i = 0; i < N; i++) {
        long d = input[i];
        long q = ((d * m) >> 52);
        d = d - q * p;
        if (d >= p) d -= p;
        if (d < 0) d += p;
        output[i] = d;
    }
}

int verify_results(long *gpu_result, long *cpu_result, long N) {
    for (long i = 0; i < N; i++) {
        if (gpu_result[i] != cpu_result[i]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    long p = 3329;
    long N = 1024; // 配列サイズ

    long *input = (long *)malloc(N * sizeof(long));
    long *gpu_output = (long *)malloc(N * sizeof(long));
    long *cpu_output = (long *)malloc(N * sizeof(long));

    if (input == NULL || gpu_output == NULL || cpu_output == NULL) {
        printf("Error allocating memory\n");
        exit(1);
    }

    for (long i = 0; i < N; i++) {
        input[i] = rand() % (p * 2); // ランダムな値を生成
    }

    double inv_p = 1.0 / (double)p;
    long m = (long)(inv_p * (double)((1ULL << 52) + 0.5));

    cl_int err;

    cl_platform_id platform_id;
    cl_uint platform_num;
    err = clGetPlatformIDs(1, &platform_id, &platform_num);
    CHECK_CL_ERROR(err, "clGetPlatformIDs");

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    CHECK_CL_ERROR(err, "clGetDeviceIDs");

    cl_context context;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateContext");

    cl_command_queue commands;
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    CHECK_CL_ERROR(err, "clCreateCommandQueue");

    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelsource, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    cl_kernel kernel;
    kernel = clCreateKernel(program, "barrett_reduction", &err);
    CHECK_CL_ERROR(err, "clCreateKernel");

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, input, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer d_input");
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(long) * N, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer d_output");

    err = clSetKernelArg(kernel, 0, sizeof(long), &p);
    CHECK_CL_ERROR(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(long), &m);
    CHECK_CL_ERROR(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_input);
    CHECK_CL_ERROR(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_output);
    CHECK_CL_ERROR(err, "clSetKernelArg 3");

    size_t global = N;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, sizeof(long) * N, gpu_output, 0, NULL, NULL);
    CHECK_CL_ERROR(err, "clEnqueueReadBuffer");

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    cpu_barrett_reduction(p, m, input, cpu_output, N);

    if (!verify_results(gpu_output, cpu_output, N)) {
        printf("Results do not match\n");
    } else {
        printf("Results match\n");
    }

    free(input);
    free(gpu_output);
    free(cpu_output);

    return 0;
}
