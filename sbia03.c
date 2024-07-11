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

int power(int a, int n) {
    if (n == 1) return a;
    else if (n == 0) return 1;
    else if (n % 2 == 0) {
        int k = power(a, n / 2);
        return (k * k);
    } else if (n % 2 == 1) {
        int k = power(a, (n - 1) / 2);
        return (k * k * a);
    }
    return 0;
}

const char *kernelsource = "__kernel void vadd (           \n" \
"   const int N,                                 \n" \
"   const int pN,                                \n" \
"   const int g,                                 \n" \
"   const int m,                                 \n" \
"   const int p,                                 \n" \
"   const int n,                                 \n" \
"   __global int *x,                             \n" \
"   __global int *x_copy,                        \n" \
"   __global int *rou,                           \n" \
"   __local int *shared_x,                       \n" \
"   __local int *shared_x_copy) {                \n" \
"       int j = get_global_id(0);                \n" \
"       int kk = 0;                              \n" \
"       int a = 1;                               \n" \
"       int b = 1 << (n-1);                 \n" \
"       for (int i = 0; i < n; i++) {            \n" \
"           int j2 = j << 1;                     \n" \
"           int j_mod_a = j % a;                 \n" \
"           int c, d;                            \n" \
"           if (i == 0){                         \n" \
"               shared_x[j] = 0;    \n" \
"               shared_x_copy[j] = x_copy[j];    \n" \
"               shared_x_copy[j + (N >> 1)] = x_copy[j + (N >> 1)]; \n" \
"           }                                    \n" \
"       if (kk == 0) {                           \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           c = shared_x_copy[j] ;              \n" \
"           if (c >= p) c -= p;                  \n" \
"           if (c < 0) c += p;                   \n" \
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * shared_x_copy[j + (N >> 1) % p]; \n" \
"          //printf(\"GPU Step kk == 0: j = %d, c = %d, d = %d, x[%d] = %d, x[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x[j2 - j_mod_a], j2 - j_mod_a + a, shared_x[j2 - j_mod_a + a]); \n" \
"           d = (d - ((d * m) >> 31) * p) ;       \n" \
"           if (d >= p) d -= p;                  \n" \
"           if (d < 0) d += p;                   \n" \
"           //printf(\"GPU Step kk == 0: j = %d, c = %d, d = %d, x[%d] = %d, x[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x[j2 - j_mod_a], j2 - j_mod_a + a, shared_x[j2 - j_mod_a + a]); \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           shared_x[j2 - j_mod_a] = (c + d) ;         \n" \
"           if (shared_x[j2 - j_mod_a] < 0) shared_x[j2 - j_mod_a] += p; \n" \
"           shared_x[j2 - j_mod_a + a] = (c + p - d) % p;     \n" \
"           if (shared_x[j2 - j_mod_a + a] < 0) shared_x[j2 - j_mod_a + a] += p; \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           //printf(\"GPU Step kk == 0: j = %d, c = %d, d = %d, x[%d] = %d, x[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x[j2 - j_mod_a], j2 - j_mod_a + a, shared_x[j2 - j_mod_a + a]); \n" \
"       } else {                                 \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           c = shared_x[j];                   \n" \
"           if (c >= p) c -= p;                  \n" \
"           if (c < 0) c += p;                   \n" \
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * shared_x[j + (N >> 1)] ; \n" \
"           //printf(\"GPU Step kk == 1: j = %d, c = %d, d = %d, x_copy[%d] = %d, x_copy[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x_copy[j2 - j_mod_a], j2 - j_mod_a + a, shared_x_copy[j2 - j_mod_a + a]); \n" \
"           d = (d - ((d * m) >> 31) * p) ;       \n" \
"           if (d >= p) d -= p;                  \n" \
"           if (d < 0) d += p;                   \n" \
"           //printf(\"GPU Step kk == 1: j = %d, c = %d, d = %d, x_copy[%d] = %d, x_copy[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x_copy[j2 - j_mod_a], j2 - j_mod_a + a, shared_x_copy[j2 - j_mod_a + a]); \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           shared_x_copy[j2 - j_mod_a] = (c + d) % p;    \n" \
"           if (shared_x_copy[j2 - j_mod_a] < 0) shared_x_copy[j2 - j_mod_a] += p; \n" \
"           shared_x_copy[j2 - j_mod_a + a] = (c - d) % p; \n" \
"           if (shared_x_copy[j2 - j_mod_a + a] < 0) shared_x_copy[j2 - j_mod_a + a] += p; \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           //printf(\"GPU Step kk == 1: j = %d, c = %d, d = %d, x_copy[%d] = %d, x_copy[%d] = %d\\n\", j, c, d, j2 - j_mod_a, shared_x_copy[j2 - j_mod_a], j2 - j_mod_a + a, shared_x_copy[j2 - j_mod_a + a]); \n" \
"       }                                        \n" \
"           if (i == n - 1){                     \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"               x_copy[j2 - j_mod_a] = shared_x_copy[j2 - j_mod_a]; \n" \
"               x_copy[j2 - j_mod_a + a] = shared_x_copy[j2 - j_mod_a + a]; \n" \
"               barrier(CLK_LOCAL_MEM_FENCE);    \n" \
"           }                                    \n" \
"           a <<= 1;                                  \n" \
"           b >>= 1;                                  \n" \
"           kk = 1 - kk;                              \n" \
"       }                                             \n" \
"}                                                    \n";
int main(int argc, char** argv) {
    int g = 17;
    int p = 3329;

    for (int n = 4; n <= 23; n++) {
        double total_time = 0.0;
        int N = 1 << n;
        for (int iii = 0; iii < 10; iii++) {
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

            cl_kernel ko_vadd;
            ko_vadd = clCreateKernel(program, "vadd", &err);
            CHECK_CL_ERROR(err, "clCreateKernel");

            int *x = (int *)malloc(N * sizeof(int));
            int *x_copy = (int *)malloc(N * sizeof(int));
            int *rou = (int *)malloc((p - 1) * sizeof(int));
            int *x_cpu = (int *)malloc(N * sizeof(int));

            if (x == NULL || x_copy == NULL || rou == NULL || x_cpu == NULL) {
                printf("Error allocating memory\n");
                exit(1);
            }

            for (int i = 0; i < N; i++) {
                x[i] = 0;
                x_copy[i] = rand() % p;
                x_cpu[i] = x_copy[i];
                //printf("Initial data: x_copy[%d] = %d\n", i, x_copy[i]);
            }

            rou[0] = 1;
            for (int i = 0; i < p - 2; i++) rou[i + 1] = (rou[i] * g) % p;

            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);

            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, x, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_x");
            cl_mem d_x_copy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, x_copy, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_x_copy");
            cl_mem d_rou = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (p - 1), rou, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_rou");

            int pN = (p - 1) / N;
            double inv_p = 1.0 / (double)p;
            int m = (int)(inv_p * (double)((1ULL << 31) + 0.5));

            err = clSetKernelArg(ko_vadd, 0, sizeof(int), &N);
            CHECK_CL_ERROR(err, "clSetKernelArg 0");
            err = clSetKernelArg(ko_vadd, 1, sizeof(int), &pN);
            CHECK_CL_ERROR(err, "clSetKernelArg 1");
            err = clSetKernelArg(ko_vadd, 2, sizeof(int), &g);
            CHECK_CL_ERROR(err, "clSetKernelArg 2");
            err = clSetKernelArg(ko_vadd, 3, sizeof(int), &m);
            CHECK_CL_ERROR(err, "clSetKernelArg 3");
            err = clSetKernelArg(ko_vadd, 4, sizeof(int), &p);
            CHECK_CL_ERROR(err, "clSetKernelArg 4");
            err = clSetKernelArg(ko_vadd, 5, sizeof(int), &n);
            CHECK_CL_ERROR(err, "clSetKernelArg 5");
            err = clSetKernelArg(ko_vadd, 6, sizeof(cl_mem), &d_x);
            CHECK_CL_ERROR(err, "clSetKernelArg 6");
            err = clSetKernelArg(ko_vadd, 7, sizeof(cl_mem), &d_x_copy);
            CHECK_CL_ERROR(err, "clSetKernelArg 7");
            err = clSetKernelArg(ko_vadd, 8, sizeof(cl_mem), &d_rou);
            CHECK_CL_ERROR(err, "clSetKernelArg 8");
            err = clSetKernelArg(ko_vadd, 9, N*sizeof(int), NULL); // ローカルメモリの割り当て
            CHECK_CL_ERROR(err, "clSetKernelArg 9");
            err = clSetKernelArg(ko_vadd, 10,  N*sizeof(int), NULL); // ローカルメモリの割り当て
            CHECK_CL_ERROR(err, "clSetKernelArg 10");

            size_t M = 1 << (n - 1);
            size_t nn = 1;

            err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &M, &nn, 0, NULL, NULL);
            CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");
            clFinish(commands);

            gettimeofday(&end_time, NULL);


            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;

            int *ans;
            if (n % 2 == 1) {
                clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0, sizeof(int) * N, x, 0, NULL, NULL);
                ans = x;  
            } else {
                clEnqueueReadBuffer(commands, d_x_copy, CL_TRUE, 0, sizeof(int) * N, x_copy, 0, NULL, NULL);
                ans = x_copy;  
            }

            // 結果の出力
            //for (int i = 0; i < N; i++) {
            //printf("index 0: GPU = %d\n", ans[i]);
            //}

            clReleaseMemObject(d_x_copy);
            clReleaseMemObject(d_x);
            clReleaseMemObject(d_rou);
            clReleaseProgram(program);
            clReleaseKernel(ko_vadd);
            clReleaseCommandQueue(commands);
            clReleaseContext(context);

            free(x);
            free(x_copy);
            free(rou);
            free(x_cpu);

        }
        printf("Elapsed time for N = %d: %f ms\n", N, total_time * 100);
    }
    return 0;
}
