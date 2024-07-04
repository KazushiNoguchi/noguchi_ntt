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

char *kernelsource = "__kernel void vadd (           \n" \
"   const long N,                                 \n" \
"   const long pN,                                 \n" \
"   const long p,                                 \n" \
"   const long a,                                 \n" \
"   const long b,                                 \n" \
"   const long g,                                 \n" \
"   __global long *x,                             \n" \
"   __global long *x_copy,                        \n" \
"   __global long *rou,                           \n" \
"   const long kk) {                              \n" \
"       long j = get_global_id(0);                \n" \
"       long j2 = j << 1;                         \n" \
"       long j_mod_a = j % a;                     \n" \
"       long c, d;                                \n" \
"       if (kk == 0) {                            \n" \
"           c = x_copy[j];                        \n" \
"           d = rou[ (j_mod_a * b * pN) % (p - 1) ] * x_copy[j + (N >> 1)]; \n" \
"           x[j2 - j_mod_a] = ((c + d) % p + p) % p; \n" \
"           x[j2 - j_mod_a + a] = ((c - d) % p + p) % p; \n" \
"       } else {                                  \n" \
"           c = x[j];                             \n" \
"           d = rou[ (j_mod_a * b * pN) % (p - 1) ] * x[j + (N >> 1)]; \n" \
"           x_copy[j2 - j_mod_a] = ((c + d) % p + p) % p; \n" \
"           x_copy[j2 - j_mod_a + a] = ((c - d) % p + p) % p; \n" \
"       }                                         \n" \
"}                                                \n" \
"\n";

int main(int argc, char** argv) {
    long g = 17;
    long p = 3329;
    for (long n = 4; n <= 23; n++) {
        double total_time = 0.0;
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
            program = clCreateProgramWithSource(context, 1, (const char **)&kernelsource, NULL, NULL);

            if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
                printf("Error building program\n");
                exit(1);
            }

            cl_kernel ko_vadd;
            ko_vadd = clCreateKernel(program, "vadd", NULL);

            long *f = (long *)malloc(N * sizeof(long));
            long *x = (long *)malloc(N * sizeof(long));
            long *x_copy = (long *)malloc(N * sizeof(long));

            if (f == NULL || x == NULL || x_copy == NULL) {
                printf("Error allocating memory\n");
                exit(1);
            }

            for (long i = 0; i < N; i++) {
                f[i] = rand() % p;
                x[i] = 0;
                x_copy[i] = 0;
            }

            long a = 1;
            long b = power(2, n - 1);

            long *rou = (long *)malloc((p - 1) * sizeof(long));
            if (rou == NULL) {
                printf("Error allocating memory\n");
                exit(1);
            }

            rou[0] = 1;
            for (long i = 0; i < p - 2; i++) rou[i + 1] = (rou[i] * g) % p;

            for (long i = 0; i < N; i++) x_copy[i] = f[i];

            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);

            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, x, NULL);
            cl_mem d_x_copy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, x_copy, NULL);
            cl_mem d_rou = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(long) * (p - 1), rou, NULL);

            long pN = (p - 1) / N;

            clSetKernelArg(ko_vadd, 0, sizeof(long), &N);
            clSetKernelArg(ko_vadd, 1, sizeof(long), &pN);
            clSetKernelArg(ko_vadd, 2, sizeof(long), &p);
            clSetKernelArg(ko_vadd, 5, sizeof(long), &g);
            clSetKernelArg(ko_vadd, 6, sizeof(cl_mem), &d_x);
            clSetKernelArg(ko_vadd, 7, sizeof(cl_mem), &d_x_copy);
            clSetKernelArg(ko_vadd, 8, sizeof(cl_mem), &d_rou);

            long kk = 0;
            size_t M = 1 << (n - 1);
            size_t nn = 1;

            for (long i = 0; i < n; i++) {
                clSetKernelArg(ko_vadd, 9, sizeof(long), &kk);
                clSetKernelArg(ko_vadd, 3, sizeof(long), &a);
                clSetKernelArg(ko_vadd, 4, sizeof(long), &b);

                clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &M, &nn, 0, NULL, NULL);
                clFinish(commands);

                a *= 2;
                b /= 2;
                kk = 1 - kk;
            }

            gettimeofday(&end_time, NULL);

            long *ans;
            if (kk == 1) {
                clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0, sizeof(long) * N, x, 0, NULL, NULL);
                ans = x;  
            } else {
                clEnqueueReadBuffer(commands, d_x_copy, CL_TRUE, 0, sizeof(long) * N, x_copy, 0, NULL, NULL);
                ans = x_copy;  
            }

            // 結果の出力
            //for (long i = 0; i < N; i++) {
            printf("index 0: GPU = %ld\n", ans[0]);
            //}

            clEnqueueReadBuffer(commands, d_x_copy, CL_TRUE, 0, sizeof(long) * N, x_copy, 0, NULL, NULL);
            clReleaseMemObject(d_x_copy);
            clReleaseMemObject(d_x);
            clReleaseMemObject(d_rou);
            clReleaseProgram(program);
            clReleaseKernel(ko_vadd);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);

            free(f);
            free(x);
            free(x_copy);
            free(rou);

            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;
        }
        printf("Elapsed time for N = %ld: %f seconds\n", N, total_time);
    }
    return 0;
}
