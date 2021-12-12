// Команда для локальной компиляции
// gcc -o omp -fopenmp omp.c -lm

#include <stdio.h>
#include <omp.h>
#include <math.h>

// Интегрируемая функция 1
long double function1(long double x) {
    return sin(2 * x) * sqrt(x);
}

// Интегрируемая функция 2
long double function2(long double x) {
   return (1.0 + pow(x, 3) * log(x));
}

int main() {
   double start_time, end_time;
   long double sum = 0.0, res, w;
   int num_threads;

   //Пределы интегрирования
   double a = 1.0, b = 5.0;

   //Число разбиений
   long n = 400000000;

   start_time = omp_get_wtime();
   w = (b - a) / (long double)n;

#pragma omp parallel shared(w) reduction(+:sum)
   {
      num_threads = omp_get_num_threads();
#pragma omp for schedule(static)
      for(long i=1; i < n - 1; i++)
         sum += function1(a + w * i);
         // sum += function2(a + w * i);
   }
   sum += (function1(a) + function1(b)) / 2.0;
   // sum += (function2(a) + function2(b)) / 2.0;
   res = w * sum;
   end_time = omp_get_wtime();
   printf("N=%ld, Nproc=%d, res=%Lf, Time=%lf \n", n, num_threads, res, end_time - start_time);
   return 0;
}
