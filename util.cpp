#include "util.h"

#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <unistd.h>

using namespace std;

extern int mpi_rank, mpi_size;  // defined in main.cpp
extern int N;                   // defined in main.cpp
extern bool V,S,W;                  // defined in main.cpp
void parse_option(int argc, char **argv){

  int opt;
  while((opt = getopt(argc, argv, "n:vswh")) != -1) {
    switch (opt) {
      case 'n': N = atoi(optarg); break;
      case 'v': V = true; break;
      case 's': S = true; break;
      case 'w': W = true; break;
      case 'h': print_help(); exit(-1); break;
      default: print_help(); exit(-1); break;
    }
  }

  if (mpi_rank == 0) {
    fprintf(stderr, "\n Model : ???\n");
    fprintf(stderr, " =============================================\n");
    fprintf(stderr, " Warming up : %s\n", W? "ON":"OFF");
    fprintf(stderr, " Validation : %s\n", V? "ON":"OFF");
    fprintf(stderr, " Save output tensor : %s\n", S? "ON":"OFF");
    fprintf(stderr, " ---------------------------------------------\n");
  }
}

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "[ERROR] Cannot open file \'%s\'\n", filename);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stderr, "[ERROR] Cannot read file \'%s\'\n", filename);
    exit(-1);
  }
  fclose(f);

  if (size != NULL) *size = (size_t)(size_ / 4);  // float
  return buf;
}

void write_binary(float *output, const char *filename, int size_) {
  fprintf(stderr, " Writing output ... ");
  fflush(stdout);
  FILE *output_fp = (FILE *)fopen(filename, "w");
  for (int i = 0; i < size_; i++) {
    fprintf(output_fp, "%04d", (int)output[i]);
  }   
  fclose(output_fp);
  fprintf(stderr, "DONE!\n");
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return 0;
  //return tv.tv_sec + tv.tv_nsec * 1.0e-9;
}


void print_help() {
  if (mpi_rank == 0) {
    fprintf(stderr, " Usage: ./translator [-n num_input_sentences] [-vpwh]\n");
    fprintf(stderr, " Options:\n");
    fprintf(stderr, "  -n : number of input sentences (default: 1)\n");
    fprintf(stderr, "  -v : validate translator.      (default: off, available input size: ~2525184)\n");
    fprintf(stderr, "  -s : save generated sentences (default: off)\n");
    fprintf(stderr, "  -w : enable warmup (default: off)\n");
    fprintf(stderr, "  -h : print this page.\n");
  }
}