#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

long power(long a, long n) {
    if (n == 1) return a;
    else if (n == 0) return 1;
    else if (n % 2 == 0) {
        long k = power(a, n / 2);
        return (k * k);
    } else if (n % 2 == 1) {
        long k = power(a, (n - 1) / 2);
        return (k * k * a);
    }
    return 0;
}

char *kernelsource = "__kernel void bit_reverse(__global long *data, const long N, __global long *reversed) { \n" \
"    long i = get_global_id(0); \n" \
"    long j = 0; \n" \
"    for (long bit = 0; bit < (long)(log2((float)N)); bit++) { \n" \
"        j = (j << 1) | (i & 1); \n" \
"        i >>= 1; \n" \
"    } \n" \
"    reversed[j] = data[get_global_id(0)]; \n" \
"} \n" \
"__kernel void cooley_tukey(__global long *data, __global long *rou, const long N, const long p, const long stage) { \n" \
"    long len = get_global_id(0); \n" \
"    long stride = 1 << stage; \n" \
"    long pos = (len / stride) * (stride * 2) + (len % stride); \n" \
"    long w = rou[(len % stride) * (N / (2 * stride))]; \n" \
"    long u = data[pos]; \n" \
"    long v = (w * data[pos + stride]) % p; \n" \
"    data[pos] = (u + v) % p; \n" \
"    data[pos + stride] = (u - v + p) % p; \n" \
"} \n";

int main(int argc, char** argv) {
    for (long n = 4; n <= 23; n++) {
        double total_time = 0.0;
        long g = 17;
        long p = 3329;
        long N = 1 << n;
        for (int iii = 0; iii < 10; iii++) {
            

            cl_platform_id platform_id;
            cl_uint platform_num;
            clGetPlatformIDs(1, &platform_id, &platform_num);

            cl_device_id device_id;
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

            cl_context context;
            context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

            cl_command_queue commands;
            commands = clCreateCommandQueue(context, device_id, 0, NULL);

            cl_program program;
            cl_int err;
            program = clCreateProgramWithSource(context, 1, (const char **)&kernelsource, NULL, &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to create program with source. Error code: %d\n", err);
                exit(1);
            }

            err = clBuildProgram(program, 0, NULL, "-cl-std=CL1.2", NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t len;
                char buffer[2048];
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                fprintf(stderr, "Error: Failed to build program executable!\n%s\n", buffer);
                exit(1);
            }

            cl_kernel ko_bit_reverse = clCreateKernel(program, "bit_reverse", NULL);
            cl_kernel ko_cooley_tukey = clCreateKernel(program, "cooley_tukey", NULL);

            long *f = (long *)malloc(N * sizeof(long));
            long *x = (long *)malloc(N * sizeof(long));
            long *rou = (long *)malloc((N / 2) * sizeof(long));

            for (long i = 0; i < N; i++) {
                f[i] = rand() % p;
                x[i] = 0;
            }

            rou[0] = 1;
            for (long i = 1; i < N / 2; i++) rou[i] = (rou[i - 1] * g) % p;

            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, x, NULL);
            cl_mem d_f = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, f, NULL);
            cl_mem d_rou = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(long) * (N / 2), rou, NULL);
            cl_mem d_reversed = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(long) * N, NULL, NULL);

            clSetKernelArg(ko_bit_reverse, 0, sizeof(cl_mem), &d_f);
            clSetKernelArg(ko_bit_reverse, 1, sizeof(long), &N);
            clSetKernelArg(ko_bit_reverse, 2, sizeof(cl_mem), &d_reversed);

            size_t global_size = N;
            size_t local_size = 1;
            clEnqueueNDRangeKernel(commands, ko_bit_reverse, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            clFinish(commands);

            clSetKernelArg(ko_cooley_tukey, 0, sizeof(cl_mem), &d_reversed);
            clSetKernelArg(ko_cooley_tukey, 1, sizeof(cl_mem), &d_rou);
            clSetKernelArg(ko_cooley_tukey, 2, sizeof(long), &N);
            clSetKernelArg(ko_cooley_tukey, 3, sizeof(long), &p);

            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);

            for (long stage = 0; stage < n; stage++) {
                clSetKernelArg(ko_cooley_tukey, 4, sizeof(long), &stage);
                global_size = N / 2; // それぞれのステージでのバタフライ計算数
                clEnqueueNDRangeKernel(commands, ko_cooley_tukey, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
                clFinish(commands);
            }

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;

            clEnqueueReadBuffer(commands, d_reversed, CL_TRUE, 0, sizeof(long) * N, x, 0, NULL, NULL);

            clReleaseMemObject(d_x);
            clReleaseMemObject(d_f);
            clReleaseMemObject(d_rou);
            clReleaseMemObject(d_reversed);
            clReleaseProgram(program);
            clReleaseKernel(ko_bit_reverse);
            clReleaseKernel(ko_cooley_tukey);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);

            free(f);
            free(x);
            free(rou);
        }

        printf("Elapsed time for N = %ld: %f seconds\n", N, total_time);
    }
    return 0;
}
