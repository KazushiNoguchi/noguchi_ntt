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
"   const int a,                                 \n" \
"   const int b,                                 \n" \
"   const int g,                                 \n" \
"   const int m,                                 \n" \
"   const int p,                                 \n" \
"   __global int *x,                             \n" \
"   __global int *x_copy,                        \n" \
"   __global int *rou,                           \n" \
"   const int kk,                                \n" \
"   __local int *shared_x,                       \n" \
"   __local int *shared_x_copy) {                \n" \
"       int j = get_global_id(0);                \n" \
"       int lid = get_local_id(0);               \n" \
"       int j2 = j << 1;                         \n" \
"       int j_mod_a = j % a;                     \n" \
"       int c, d;                                \n" \
"       if (kk == 0) {                           \n" \
"           shared_x_copy[lid] = x_copy[j];      \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           c = shared_x_copy[lid];              \n" \
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * shared_x_copy[lid + (N >> 1)]; \n" \
"           d = (d - ((d * m) >> 52) * p);       \n" \
"           if (d >= p) d -= p;                  \n" \
"           if (d < 0) d += p;                   \n" \
"           shared_x[lid] = (c + d) % p;         \n" \
"           if (shared_x[lid] < 0) shared_x[lid] += p; \n" \
"           shared_x[lid + a] = (c - d) % p;     \n" \
"           if (shared_x[lid + a] < 0) shared_x[lid + a] += p; \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           x[j2 - j_mod_a] = shared_x[lid];     \n" \
"           x[j2 - j_mod_a + a] = shared_x[lid + a]; \n" \
"       } else {                                 \n" \
"           shared_x[lid] = x[j];                \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           c = shared_x[lid];                   \n" \
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * shared_x[lid + (N >> 1)]; \n" \
"           d = (d - ((d * m) >> 52) * p);       \n" \
"           if (d >= p) d -= p;                  \n" \
"           if (d < 0) d += p;                   \n" \
"           shared_x_copy[lid] = (c + d) % p;    \n" \
"           if (shared_x_copy[lid] < 0) shared_x_copy[lid] += p; \n" \
"           shared_x_copy[lid + a] = (c - d) % p; \n" \
"           if (shared_x_copy[lid + a] < 0) shared_x_copy[lid + a] += p; \n" \
"           barrier(CLK_LOCAL_MEM_FENCE);        \n" \
"           x_copy[j2 - j_mod_a] = shared_x_copy[lid]; \n" \
"           x_copy[j2 - j_mod_a + a] = shared_x_copy[lid + a]; \n" \
"       }                                        \n" \
"}                                               \n";



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
            int a = 1;
            int b = power(2, n - 1);
            double inv_p = 1.0 / (double)p;
            int m = (int)(inv_p * (double)((1ULL << 52) + 0.5));

            err = clSetKernelArg(ko_vadd, 0, sizeof(int), &N);
            CHECK_CL_ERROR(err, "clSetKernelArg 0");
            err = clSetKernelArg(ko_vadd, 1, sizeof(int), &pN);
            CHECK_CL_ERROR(err, "clSetKernelArg 1");
            err = clSetKernelArg(ko_vadd, 2, sizeof(int), &a);
            CHECK_CL_ERROR(err, "clSetKernelArg 2");
            err = clSetKernelArg(ko_vadd, 3, sizeof(int), &b);
            CHECK_CL_ERROR(err, "clSetKernelArg 3");
            err = clSetKernelArg(ko_vadd, 4, sizeof(int), &g);
            CHECK_CL_ERROR(err, "clSetKernelArg 4");
            err = clSetKernelArg(ko_vadd, 5, sizeof(int), &m);
            CHECK_CL_ERROR(err, "clSetKernelArg 5");
            err = clSetKernelArg(ko_vadd, 6, sizeof(int), &p);
            CHECK_CL_ERROR(err, "clSetKernelArg 6");
            err = clSetKernelArg(ko_vadd, 7, sizeof(cl_mem), &d_x);
            CHECK_CL_ERROR(err, "clSetKernelArg 7");
            err = clSetKernelArg(ko_vadd, 8, sizeof(cl_mem), &d_x_copy);
            CHECK_CL_ERROR(err, "clSetKernelArg 8");
            err = clSetKernelArg(ko_vadd, 9, sizeof(cl_mem), &d_rou);
            CHECK_CL_ERROR(err, "clSetKernelArg 9");

            size_t M = 1 << (n - 1);
            int kk = 0;

            for (int i = 0; i < n; i++) {

                err = clSetKernelArg(ko_vadd, 10, sizeof(int), &kk);
                CHECK_CL_ERROR(err, "clSetKernelArg 10");
                err = clSetKernelArg(ko_vadd, 11, sizeof(cl_mem), NULL); // ローカルメモリの割り当て
                CHECK_CL_ERROR(err, "clSetKernelArg 11");
                err = clSetKernelArg(ko_vadd, 12, sizeof(cl_mem), NULL); // ローカルメモリの割り当て
                CHECK_CL_ERROR(err, "clSetKernelArg 12");

                err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &M, NULL, 0, NULL, NULL);
                CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");
                clFinish(commands);

                a <<= 1;
                b >>= 1;
                kk = 1 - kk;
            }



            gettimeofday(&end_time, NULL);

            int *ans;
            if (kk == 1) {
                clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0, sizeof(int) * N, x, 0, NULL, NULL);
                ans = x;  
            } else {
                clEnqueueReadBuffer(commands, d_x_copy, CL_TRUE, 0, sizeof(int) * N, x_copy, 0, NULL, NULL);
                ans = x_copy;  
            }

            // 結果の出力
            //for (int i = 0; i < N; i++) {
            printf("index 0: GPU = %d\n", ans[0]);
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

            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;
        }
        printf("Elapsed time for N = %d: %f seconds\n", N, total_time);
    }
    return 0;
}
