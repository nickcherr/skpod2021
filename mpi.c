// Команда для локальной компиляции
// mpicc -o mpi mpi.c -lm

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Интегрируемая функция 1
long double function1(long double x) {
  return sin(2 * x) * sqrt(x);
}

// Интегрируемая функция 2
long double function2(long double x) {
  return (1.0 + pow(x, 3) * log(x));
}

int main(int argc, char** argv) {
  double start_time, end_time;
  long double sum = 0.0, all_sum, res, w;

  int error_code = MPI_Init(&argc, &argv);
  if (error_code != 0)
    return error_code;

  //Пределы интегрирования
  long double a = 1.0, b = 5.0;

  //Число разбиений
  long n = 100000000;

  //Переменные для группового взаимодействия процессов в MPI
  int myrank, ranksize, i;

  //Определяем свой номер в группе:
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  //Определяем размер группы:
  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

  start_time = MPI_Wtime();
  w = (b - a) / (long double)n;

  long double cur_a, cur_b, cur_w;
  long num_per_p;
  long double *sbuf = NULL;

  if (!myrank) { //Процесс-Master
    //Определяем размер диапазона для каждого процесса:
    num_per_p = n / ranksize;
    sbuf = (long double*)calloc(ranksize*3, sizeof(long double));

    if (sbuf == NULL) return 1; 

    cur_a = a;
    for (i = 0; i < n - ranksize * num_per_p; i++) {
      cur_b = cur_a + (num_per_p + 1) * w;
      sbuf[i * 3] = cur_a;
      sbuf[i * 3 + 1] = cur_b;
      sbuf[i * 3 + 2] = w;
      cur_a = cur_b;
    }

    for (i = n - ranksize * num_per_p; i < ranksize; i++) {
      cur_b = cur_a + num_per_p * w;
      sbuf[i * 3] = cur_a;
      sbuf[i * 3 + 1] = cur_b;
      sbuf[i * 3 + 2] = w;
      cur_a = cur_b;
    }
  }

  long double rbuf[3];

  //Рассылка всем процессам, включая процесс-мастер
  //начальных данных для расчета:
  MPI_Scatter(sbuf, 3, MPI_LONG_DOUBLE, rbuf, 3, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
  if (!myrank)
    if(sbuf)
      free(sbuf);

  cur_a = rbuf[0];
  cur_b = rbuf[1];
  cur_w = rbuf[2];

  sum = 0.0;

  //Расчет интеграла в своем диапазоне, выполняют все процессы:
  // sum += (function1(cur_a) + function1(cur_b)) / 2.0 ;
  sum += (function2(cur_a) + function2(cur_b)) / 2.0 ;
  cur_a += cur_w;

  for( ; cur_a < cur_b; cur_a += cur_w)
    // sum += function1(cur_a);
    sum += function2(cur_a);

  //Редуцируем значения интегралов от процессов:
  MPI_Reduce(&sum, &all_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (!myrank) { //Процесс-Master
    res = w * all_sum;
    end_time = MPI_Wtime();
    printf("N=%ld, Nproc=%d, res=%Lf, Time=%lf \n", n, ranksize, res, end_time - start_time);
  }

  //Завершение работы с MPI
  MPI_Finalize();

  return 0;
}
