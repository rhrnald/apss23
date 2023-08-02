#pragma once

#include "tensor.h"

void initialize_model_seq(const char *parameter_fname);
void model_forward_seq(Tensor *input, Tensor *output);
void finalize_model_seq();
