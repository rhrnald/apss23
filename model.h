#pragma once

#include "tensor.h"

void initialize_model(const char *parameter_fname);
void model_forward(Tensor *input, Tensor *output);
void finalize_model();