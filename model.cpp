#include <math.h>

#include "model.h"
#include "util.h"
extern int N;

// class BrainTumorModel(nn.Module):
//  
//  def __init__(self):
//      super().__init__()
//      self.conv0 = nn.Sequential(
//          nn.Conv2d(1,128,kernel_size=3),
//          nn.InstanceNorm2d(128),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//      
//      self.conv1 = nn.Sequential(
//          nn.Conv2d(128,256,kernel_size=3),
//          nn.InstanceNorm2d(256),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.linear1 = nn.Linear(62,128)
//      self.linear2 = nn.Linear(128,64)
//      self.flat = nn.Flatten(1)
//      self.linear3 = nn.Linear(1015808,2)
//
//  def forward(self,x):
//      x = self.conv0(x)
//      x = self.conv1(x)
//      x = F.relu(self.linear1(x))
//      x = self.linear2(x)
//      x = self.flat(x)
//      x = self.linear3(x)
//
//      return x

static float *conv0_weight, *conv0_bias, *conv1_weight, *conv1_bias,
             *linear1_weight, *linear1_bias,
             *linear2_weight, *linear2_bias,
             *linear3_weight, *linear3_bias,
             *instanceNorm2d0_weight, *instanceNorm2d0_bias,
             *instanceNorm2d1_weight, *instanceNorm2d1_bias;

static Tensor *c1, *i1, *m1, *c2, *i2, *m2, *l1, *l2;
void initialize_model(const char* parameter_fname){
  size_t m; //2345922
  float *buf=(float*)read_binary(parameter_fname, &m);
  conv0_weight         = buf; buf+=1152;//1152
  conv0_bias           = buf; buf+=128;
  instanceNorm2d0_weight  = buf; buf+=128;
  instanceNorm2d0_bias    = buf; buf+=128;
  conv1_weight         = buf; buf+=294912;
  conv1_bias           = buf; buf+=256;
  instanceNorm2d1_weight  = buf; buf+=256;
  instanceNorm2d1_bias    = buf; buf+=256;
  linear1_weight         = buf; buf+=7936;
  linear1_bias           = buf; buf+=128;
  linear2_weight         = buf; buf+=8192;
  linear2_bias           = buf; buf+=64;
  linear3_weight         = buf; buf+=2031616;
  linear3_bias           = buf; buf+=2;
	
  c1 = new Tensor(N,128,254,254);
  i1 = new Tensor(N,128,254,254);
  m1 = new Tensor(N, 128, 127, 127);
  c2 = new Tensor(N, 256, 125, 125);
  i2 = new Tensor(N, 256, 125, 125);
  m2 = new Tensor(N, 256, 62, 62);
  l1 = new Tensor(N, 256, 62, 128);
  l2 = new Tensor(N, 256, 62, 64);
}
// Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// Size of in  = N * C_IN * H_IN * W_IN
// Size of out = N * C_OUT * (H_IN-K+1) * (W_IN-K+1)
// Weight : C_OUT * C_IN * K * K
// Bias : C_OUT
static void conv2d(float *in, float *out, float *weight, float *bias, int N, int C_IN, int H_IN, int W_IN, int K, int C_OUT);

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d(float *in, float *out, int N, int H_IN, int W_IN, int kH, int kW);

// InstanceNorm2D
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
// size of in  = N * C * H * W
// size of out = N * C * H * W
// weight : C
// bias : C
static void instancenorm2d(float *in, float *out, float *weight, float *bias, int N, int C, int H, int W);

// Linear
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// size of in  = N * H_IN
// size of out = N * H_OUT
// weight : H_OUT * H_IN
// bias : H_OUT
static void linear(float *in, float *out, float *weight, float *bias, int N, int H_IN, int H_OUT);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(float *inout, int N);

void model_forward(Tensor *input, Tensor *output){
  conv2d(input->buf, c1->buf, conv0_weight, conv0_bias, N, 1, 128, 256, 256, 3);
  instancenorm2d(c1->buf, i1->buf, instanceNorm2d0_weight, instanceNorm2d0_bias, N, 128, 254, 254);
  maxpool2d(i1->buf, m1->buf, N*128, 254, 254, 2, 2);
  relu(m1->buf, N*128*127*127);

  conv2d(m1->buf, c2->buf, conv1_weight, conv1_bias, N, 128, 256, 127, 127, 3);
  instancenorm2d(c2->buf, i2->buf, instanceNorm2d1_weight, instanceNorm2d1_bias, N, 256, 125, 125);
  maxpool2d(i2->buf, m2->buf, N*256, 125, 125, 2, 2);
  relu(m2->buf, N*256*62*62);

  linear(m2->buf, l1->buf, linear1_weight, linear1_bias, N*256*62, 62, 128);
  relu(l1->buf, N*256*62*128);
  linear(l1->buf, l2->buf, linear2_weight, linear2_bias, N*256*62, 128, 64);
  l2->reshape({N, 1015808});
  linear(l2->buf, output->buf, linear3_weight, linear3_bias, N, 1015808, 2);
}


static void conv2d(float *in, float *out, float *weight, float *bias, int N, int C_IN, int C_OUT, int H_IN, int W_IN, int K) {
  int H_OUT = H_IN -K + 1, W_OUT = W_IN - K + 1;
  for(int n = 0; n < N; n++) {
    for(int c_out = 0; c_out < C_OUT; c_out++) {
      for (int h_out = 0; h_out < H_OUT; h_out++) {
        for (int w_out = 0; w_out < W_OUT; w_out++) {
          out[n*C_OUT*H_OUT*W_OUT + c_out*H_OUT*W_OUT + h_out*W_OUT + w_out] = bias[c_out];
          for(int c_in = 0; c_in < C_IN; c_in++) {
            for(int kh = 0 ; kh < K; kh++) {
              for(int kw = 0 ; kw < K; kw++) {
                out[n*C_OUT*H_OUT*W_OUT + c_out*H_OUT*W_OUT + h_out*W_OUT + w_out] += in[n*C_IN*H_IN*W_IN + c_in*H_IN*W_IN +(h_out+kh)*W_IN + (w_out+kw)] *
                                                                                    weight[c_out*C_IN*K*K + c_in*K*K + kh*K + kw];
              }    
            }    
          }
        }
      }
    }
  }
}

static void instancenorm2d(float *in, float *out, float *weight, float *bias, int N, int C, int H, int W) {
  for(int n=0; n<N; n++) {
    for(int c=0; c<C; c++) {
      float e=0,v=0;

	  // Caculate mean
      for(int h=0; h<H; h++) {
        for(int w=0; w<W; w++) {
          e += in[n*C*H*W + c*H*W + h*W + w];
        }
      }
      e/=H*W;
	  
	  // Caculate Variance
      for(int h=0; h<H; h++) {
        for(int w=0; w<W; w++) {
          v += (in[n*C*H*W + c*H*W + h*W + w]-e) * (in[n*C*H*W + c*H*W + h*W + w]-e);
        }
      }
	  v/=H*W;

      for(int h=0; h<H; h++) {
        for(int w=0; w<W; w++) {
          out[n*C*H*W + c*H*W + h*W + w] = (in[n*C*H*W + c*H*W + h*W + w]-e)/sqrt(v+1e-5) * weight[c] + bias[c];
        }
      }
    }
  }
}



static void linear(float *in, float *out, float *weight, float *bias, int N, int H_IN, int H_OUT) {
  for(int n = 0; n < N; n++) {
    for(int h_out = 0; h_out < H_OUT; h_out++) {
      out[n*H_OUT + h_out] = bias[h_out];
      for(int h_in = 0 ; h_in < H_IN; h_in++) {
        out[n*H_OUT + h_out] += in[n*H_IN + h_in] * weight[h_out*H_IN + h_in];
      }
    }
  }
}


static void maxpool2d(float *in, float *out, int N, int H_IN, int W_IN, int kH, int kW) {
  int H_OUT = H_IN/kH;
  int W_OUT = W_IN/kW;

  for (int n = 0; n < N; n++) {
    for(int h_out = 0; h_out < H_OUT; h_out++) {
      for(int w_out = 0; w_out < W_OUT; w_out++) {
        out[n*H_OUT*W_OUT + h_out*W_OUT + w_out] = 0; //어쩌피 ReLU
        //out[n*H_OUT*W_OUT + h_out*W_OUT + w_out] = in[n*H_IN*W_IN + (h_out*kH)*H_IN + (w_out*kW)]; 
        for(int kh = 0; kh < kH; kh++)
          for(int kw = 0; kw < kW; kw++)            
            out[n*H_OUT*W_OUT + h_out*W_OUT + w_out] = fmaxf(out[n*H_OUT*W_OUT + h_out*W_OUT + w_out], in[n*H_IN*W_IN + (h_out*kH + kh)*H_IN + (w_out*kW + kw)]);
      }
    }
  }
}

static void relu(float *inout, int N) {
  for (int n = 0; n < N; n++) {
    inout[n] = fmaxf(inout[n], 0);
  }
}

void finalize_model() {
  delete(c1);
  delete(i1);
  delete(m1);
  delete(c2);
  delete(i2);
  delete(m2);
  delete(l1);
  delete(l2);
}
