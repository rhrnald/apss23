//todo:
//read binary
//forward
//아웃풋 저장


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>

#include "util.h"
#include "tensor.h"
#include "model.h"

int mpi_rank=0, mpi_size;
int N;
int S,W,V;
static const char parameter_fname[30] = "./data/weights.bin";
static const char input_fname[30] = "./data/inputTensor.bin";
static const char output_fname[30] = "./output.bin"; // TODO write your student id

int main(int argc, char **argv) {
  //MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  //MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  parse_option(argc, argv);
  Tensor *input, *output;
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  // Initilization and Reading inputs must be done in this block.   //
  // A node with mpi_rank 0 will read all data.              //
  ////////////////////////////////////////////////////////////////////
  // parameter를 미리 옮길까 말까                                

  
  if (mpi_rank == 0) {

    //Get input from binary file
    input = new Tensor(input_fname);
    if(input->get_elem() % (1<<16) != 0) {
      fprintf(stderr, " Wrong input tensor shape: %d\n", input->get_elem());
    }

    N = input->get_elem() >> 16;
    input->reshape({N, 1, 256, 256});

    
    //Initalize model
    initialize_model(parameter_fname);

    //Define output Tensor
    output = new Tensor({N,2});
  }

  if(W) {
    if (mpi_rank == 0) {
      fprintf(stderr, " Warming up... \n");

      for(int i=0; i<W; i++) {
        model_forward(input, output);
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////
  // COMMUNICATION & COMPUTATION                                    //
  // All communication and computation must be done in this block.  //
  // It is free to use any number of nodes and gpus.                //
  ////////////////////////////////////////////////////////////////////

  double st, et;
  if (mpi_rank == 0) {
    fprintf(stderr, " Start... \n");

    st = get_time(); 
  }
  
  //MPI_Barrier(MPI_COMM_WORLD);
  model_forward(input, output);
  //MPI_Barrier(MPI_COMM_WORLD);

  
  if (mpi_rank == 0) {
    et = get_time();
    fprintf(stderr, "DONE!\n");
    fprintf(stderr, " ---------------------------------------------\n");
    fprintf(stderr, " Elapsed time : %lf s\n", et-st);
    fprintf(stderr, " Throughput   : %lf GFLOPS\n", 2982*(double)N/(et-st));
  }

  if(S) {
    fprintf(stderr, " Saving output...\n");
    output->save(output_fname);
  }
  
  if(V) {

  }

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////
  
  //MPI_Finalize();

  return EXIT_SUCCESS;
}
