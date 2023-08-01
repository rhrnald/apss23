#pragma once
#include <vector>
#include <stdio.h>
using namespace std;

struct Tensor {
  int n=0;
  int ndim = 0;
  int shape[4];
  float *buf = nullptr;
  Tensor(const int d1, const int d2, const int d3, const int d4);
  Tensor(const vector<int> &shape_);
  Tensor(const char* filename);

  ~Tensor();
  
  void save(const char* filename); 
  int get_elem();
  void reshape(const vector<int> &shape_);
};