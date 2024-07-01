#include <bits/stdc++.h>
#include <vector>
#include <random>
#include <sys/time.h>
#include <iostream>

using namespace std;

long primitive_root(long p) {
    bool flag = false;
    long g;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(2, p - 1);

    while (!flag) {
        long k = 1;
        g = dist(gen);
        flag = true;
        for (long i = 1; i <= p - 2; i++) {
            k = (k * g) % p;
            if (k == 1) flag = false;
        }
    }

    return g;
}

long power_mod(long a, long n, long p) {
    if (n == 1) return a;
    else if (n == 0) return 1;
    else if (n % 2 == 0) {
        long k = power_mod(a, n / 2, p) % p;
        return (k * k) % p;
    } else if (n % 2 == 1) {
        long k = power_mod(a, (n - 1) / 2, p) % p;
        return (k * k * a) % p;
    }
    return 0;
}

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

long inverse(long x, long p) {
    return power_mod(x, p - 2, p);
}

int main() {
    for (long n = 4; n <= 23; n++) {
        double total_time = 0.0;
        const long N = power(2, n);
        long g = 17;
        long p = 3329;

        // 配列のメモリ割り当て
        long *ml = new long[p - 1];

        ml[0] = 1;
        for (long i = 0; i < p - 2; i++) ml[i + 1] = (ml[i] * g) % p;

        for (long kkk = 0; kkk < 10; kkk++) {
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dist(0, p);

            long *f = new long[N];
            long *x = new long[N];
            long *x_copy = new long[N];

            for (long i = 0; i < N; i++) f[i] = dist(gen);
            for (long i = 0; i < N; i++) {
                x[i] = 0;
                x_copy[i] = f[i];
            }

            long a = 1;
            long b = power(2, n - 1);
            long c, d;
            long imod2 = 0;
            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);
            for (long i = 0; i < n; i++) {
                if (imod2 == 0) {
                    for (long j = 0; j < b; j++) {
                        for (long k = 0; k < a; k++) {
                            c = x_copy[a * j + k];
                            d = ml[(k * b * (p - 1) / N) % (p - 1)] * x_copy[a * j + N / 2 + k];
                            x[2 * a * j + k] = ((c + d) % p + p) % p;
                            x[2 * a * j + k + a] = ((c - d) % p + p) % p;
                        }
                    }
                } else {
                    for (long j = 0; j < b; j++) {
                        for (long k = 0; k < a; k++) {
                            c = x[a * j + k];
                            d = ml[(k * b * (p - 1) / N) % (p - 1)] * x[a * j + N / 2 + k];
                            x_copy[2 * a * j + k] = ((c + d) % p + p) % p;
                            x_copy[2 * a * j + k + a] = ((c - d) % p + p) % p;
                        }
                    }
                }
                a = a * 2;
                b = b / 2;
                imod2 = 1 - imod2;
            }

            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;

            delete[] f;
            delete[] x;
            delete[] x_copy;
        }

        delete[] ml; // メモリ解放

        cout << "Elapsed time for N = " << N << ": " << total_time << " seconds" << endl;
    }
    return 0;
}
