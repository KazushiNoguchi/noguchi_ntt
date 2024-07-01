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

const char *kernelsource = "__kernel void vadd (           \n" \
"   const long N,                                 \n" \
"   const long pN,                                \n" \
"   const long a,                                 \n" \
"   const long b,                                 \n" \
"   const long g,                                 \n" \
"   const long m,                                 \n" \
"   const long p,                                 \n" \
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
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * x_copy[j + (N >> 1)]; \n" \
"           long q = ((d * m) >> 52);             \n" \
"           d = d - q * p;                        \n" \
"           if (d >= p) d -= p;                   \n" \
"           if (d < 0) d += p;                    \n" \
"           x[j2 - j_mod_a] = (c + d) % p;        \n" \
"           if (x[j2 - j_mod_a] < 0) x[j2 - j_mod_a] += p; \n" \
"           x[j2 - j_mod_a + a] = (c - d) % p;    \n" \
"           if (x[j2 - j_mod_a + a] < 0) x[j2 - j_mod_a + a] += p; \n" \
"       } else {                                  \n" \
"           c = x[j];                             \n" \
"           d = rou[(j_mod_a * b * pN) % (p - 1)] * x[j + (N >> 1)]; \n" \
"           long q = ((d * m) >> 52);             \n" \
"           d = d - q * p;                        \n" \
"           if (d >= p) d -= p;                   \n" \
"           if (d < 0) d += p;                    \n" \
"           x_copy[j2 - j_mod_a] = (c + d) % p;   \n" \
"           if (x_copy[j2 - j_mod_a] < 0) x_copy[j2 - j_mod_a] += p; \n" \
"           x_copy[j2 - j_mod_a + a] = (c - d) % p; \n" \
"           if (x_copy[j2 - j_mod_a + a] < 0) x_copy[j2 - j_mod_a + a] += p; \n" \
"       }                                         \n" \
"}                                                \n";

// CPUでの計算関数
void cpu_vadd(long N, long pN, long a, long b, long g, long m, long p, long *x, long *x_copy, long *rou, long kk) {
    for (long j = 0; j < N / 2; j++) {
        long j2 = j << 1;
        long j_mod_a = j % a;
        long c, d;
        if (kk == 0) {
            c = x_copy[j];
            d = rou[(j_mod_a * b * pN) % (p - 1)] * x_copy[j + (N >> 1)];
            long q = ((d * m) >> 52);
            d = d - q * p;
            if (d >= p) d -= p;
            if (d < 0) d += p;
            x[j2 - j_mod_a] = (c + d) % p;
            if (x[j2 - j_mod_a] < 0) x[j2 - j_mod_a] += p;
            x[j2 - j_mod_a + a] = (c - d) % p;
            if (x[j2 - j_mod_a + a] < 0) x[j2 - j_mod_a + a] += p;
        } else {
            c = x[j];
            d = rou[(j_mod_a * b * pN) % (p - 1)] * x[j + (N >> 1)];
            long q = ((d * m) >> 52);
            d = d - q * p;
            if (d >= p) d -= p;
            if (d < 0) d += p;
            x_copy[j2 - j_mod_a] = (c + d) % p;
            if (x_copy[j2 - j_mod_a] < 0) x_copy[j2 - j_mod_a] += p;
            x_copy[j2 - j_mod_a + a] = (c - d) % p;
            if (x_copy[j2 - j_mod_a + a] < 0) x_copy[j2 - j_mod_a + a] += p;
        }
    }
}

// 結果の比較関数
int verify_results(long *gpu_result, long *cpu_result, long N) {
    for (long i = 0; i < N; i++) {
        if (gpu_result[i] != cpu_result[i]) {
            printf("Mismatch at index %ld: GPU result = %ld, CPU result = %ld\n", i, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    long g = 17;
    long p = 3329;

    for (long n = 4; n <= 23; n++) {
        double total_time = 0.0;
        long N = 1 << n;
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

            long *x = (long *)malloc(N * sizeof(long));
            long *x_copy = (long *)malloc(N * sizeof(long));
            long *rou = (long *)malloc((p - 1) * sizeof(long));
            long *cpu_x = (long *)malloc(N * sizeof(long));
            long *cpu_x_copy = (long *)malloc(N * sizeof(long));
            long *cpu_rou = (long *)malloc((p - 1) * sizeof(long));

            if (x == NULL || x_copy == NULL || rou == NULL || cpu_x == NULL || cpu_x_copy == NULL || cpu_rou == NULL) {
                printf("Error allocating memory\n");
                exit(1);
            }

            for (long i = 0; i < N; i++) {
                x[i] = 0;
                x_copy[i] = rand() % p;
                cpu_x[i] = 0;
                cpu_x_copy[i] = x_copy[i];
            }

            // rouテーブルをCPU側で計算
            rou[0] = 1;
            for (long i = 0; i < p - 2; i++) rou[i + 1] = (rou[i] * g) % p;

            // rouテーブルをGPU側で計算
            cpu_rou[0] = 1;
            for (long i = 0; i < p - 2; i++) cpu_rou[i + 1] = (cpu_rou[i] * g) % p;

            // CPUとGPUのrouテーブルを比較
            for (long i = 0; i < p - 1; i++) {
                if (rou[i] != cpu_rou[i]) {
                    printf("Mismatch in rou table at index %ld: GPU rou = %ld, CPU rou = %ld\n", i, rou[i], cpu_rou[i]);
                    exit(1);
                }
            }

            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);

            cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, x, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_x");
            cl_mem d_x_copy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(long) * N, x_copy, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_x_copy");
            cl_mem d_rou = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(long) * (p - 1), rou, &err);
            CHECK_CL_ERROR(err, "clCreateBuffer d_rou");

            long pN = (p - 1) / N;
            long a = 1 << (n - 1);  // aは2のべき乗
            long b = power(2, n - 1);  // bも2のべき乗
            double inv_p = 1.0 / (double)p;
            long m = (long)(inv_p * (double)((1ULL << 52) + 0.5));

            err = clSetKernelArg(ko_vadd, 0, sizeof(long), &N);
            CHECK_CL_ERROR(err, "clSetKernelArg 0");
            err = clSetKernelArg(ko_vadd, 1, sizeof(long), &pN);
            CHECK_CL_ERROR(err, "clSetKernelArg 1");
            err = clSetKernelArg(ko_vadd, 2, sizeof(long), &a);
            CHECK_CL_ERROR(err, "clSetKernelArg 2");
            err = clSetKernelArg(ko_vadd, 3, sizeof(long), &b);
            CHECK_CL_ERROR(err, "clSetKernelArg 3");
            err = clSetKernelArg(ko_vadd, 4, sizeof(long), &g);
            CHECK_CL_ERROR(err, "clSetKernelArg 4");
            err = clSetKernelArg(ko_vadd, 5, sizeof(long), &m);
            CHECK_CL_ERROR(err, "clSetKernelArg 5");
            err = clSetKernelArg(ko_vadd, 6, sizeof(long), &p);
            CHECK_CL_ERROR(err, "clSetKernelArg 6");
            err = clSetKernelArg(ko_vadd, 7, sizeof(cl_mem), &d_x);
            CHECK_CL_ERROR(err, "clSetKernelArg 7");
            err = clSetKernelArg(ko_vadd, 8, sizeof(cl_mem), &d_x_copy);
            CHECK_CL_ERROR(err, "clSetKernelArg 8");
            err = clSetKernelArg(ko_vadd, 9, sizeof(cl_mem), &d_rou);
            CHECK_CL_ERROR(err, "clSetKernelArg 9");

            long kk = 0;

            size_t M = 1 << (n - 1);
            size_t nn = 1;

            for (long i = 0; i < n; i++) {
                err = clSetKernelArg(ko_vadd, 10, sizeof(long), &kk);
                CHECK_CL_ERROR(err, "clSetKernelArg 10");

                err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &M, &nn, 0, NULL, NULL);
                CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");
                clFinish(commands);

                // CPUでも同様の計算を実行
                cpu_vadd(N, pN, a, b, g, m, p, cpu_x, cpu_x_copy, cpu_rou, kk);

                M >>= 1; // Mを半分に
                kk = 1 - kk;

                // ステップごとのGPUとCPUの結果を比較
                if (kk == 0) {
                    err = clEnqueueReadBuffer(commands, d_x, CL_TRUE, 0, sizeof(long) * N, x, 0, NULL, NULL);
                    CHECK_CL_ERROR(err, "clEnqueueReadBuffer");
                    if (!verify_results(x, cpu_x, N)) {
                        printf("Mismatch after step %ld for N = %ld\n", i, N);
                        exit(1);
                    }
                } else {
                    err = clEnqueueReadBuffer(commands, d_x_copy, CL_TRUE, 0, sizeof(long) * N, x_copy, 0, NULL, NULL);
                    CHECK_CL_ERROR(err, "clEnqueueReadBuffer");
                    if (!verify_results(x_copy, cpu_x_copy, N)) {
                        printf("Mismatch after step %ld for N = %ld\n", i, N);
                        exit(1);
                    }
                }
            }

            gettimeofday(&end_time, NULL);

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
            free(cpu_x);
            free(cpu_x_copy);
            free(cpu_rou);

            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;
        }
        printf("Elapsed time for N = %ld: %f seconds\n", N, total_time);
    }
    return 0;
}
