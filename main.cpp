#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "util.h"
#include "tensor.h"
#include "model.h"

int N;
int S,W,V;

char parameter_fname[30] = "./data/weights.bin";
char input_fname[30] = "./data/sample_input.bin";
char answer_fname[30] = "./data/sample_answer.bin"; // TODO write your student id
char output_fname[30] = "./output.bin"; // TODO write your student id

int main(int argc, char **argv) {
  //MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  //MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  parse_option(argc, argv);
  Tensor *input, *output;
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  // Initilization and Reading inputs must be done in this block.   //
  ////////////////////////////////////////////////////////////////////

  
  //Get input from binary file
  input = new Tensor(input_fname);
  if(input->get_elem() % (1<<16) != 0) {
    fprintf(stderr, " Wrong input tensor shape: %d\n", input->get_elem());
  }

  N = input->get_elem() >> 16;
  input->reshape({N, 1, 256, 256});
  
    
  //Define output Tensor
  output = new Tensor({N,2});

  //Initalize model
  initialize_model(parameter_fname);

  //Warmup
  if(W) {
    fprintf(stderr, " Warming up... \n");

    for(int i=0; i<W; i++) {
      model_forward(input, output);
    }
  }
  
  ////////////////////////////////////////////////////////////////////
  // COMMUNICATION & COMPUTATION                                    //
  // All communication and computation must be done in this block.  //
  // It is free to use any number of nodes and gpus.                //
  ////////////////////////////////////////////////////////////////////

  double st=0.0, et=0.0;
  fprintf(stderr, " Start... \n");

  st = get_time(); 
  
  //MPI_Barrier(MPI_COMM_WORLD);
  model_forward(input, output);
  //MPI_Barrier(MPI_COMM_WORLD);

  
  et = get_time();
  fprintf(stderr, "  DONE!\n");
  fprintf(stderr, " ---------------------------------------------\n");
  fprintf(stderr, " Elapsed time : %lf s\n", et-st);
  fprintf(stderr, " Throughput   : %lf GFLOPS\n", 1*(double)N/(et-st)); //계산필요!

  if(S) {
    fprintf(stderr, " Saving output...\n");
    output->save(output_fname);
  }
  

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////
  
  finalize_model();
  //MPI_Finalize();

  if(V) {
      Tensor answer = Tensor(answer_fname);

      int diff=-1;
      for(int i=0; i<N*2; i++) {
	    if(abs(output->buf[i]-answer.buf[i])>1e-3) {diff=i; break;}
      }
    if(diff<0) fprintf(stderr, " Validation success!\n");
    else fprintf(stderr, " Validation fail: First mistmatch on index %d(output[i]=%f , answer[i]=%f)\n", diff, output->buf[diff], answer.buf[diff]);
  }

  return EXIT_SUCCESS;
}
