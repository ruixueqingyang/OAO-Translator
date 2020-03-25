#include <malloc.h>
#include "RunTime.h"
STATE_CONSTR StConstrTarget;
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int main( int argc, char** argv);

//#define NUM_THREAD 4
#define OPEN


FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    int no_of_nodes = 0;
    int edge_list_size = 0;
    char *input_f;
    int num_omp_threads;
	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
   
	// allocate host memory
	Node* h_graph_nodes = (Node*) OAOMalloc(sizeof(Node)*no_of_nodes); int h_graph_nodes__starting; int h_graph_nodes__no_of_edges;
	bool *h_graph_mask = (bool*) OAOMalloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) OAOMalloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) OAOMalloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	
	OAODataTrans( h_graph_mask, StConstrTarget.init(7, 3) );
	OAODataTrans( h_updating_graph_mask, StConstrTarget.init(7, 3) );
	OAODataTrans( h_graph_visited, StConstrTarget.init(7, 3) );
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	
	OAODataTrans( h_graph_mask, StConstrTarget.init(7, 3) );
	OAODataTrans( h_graph_visited, StConstrTarget.init(7, 3) );
	fscanf(fp,"%d",&source);
	// source=0; //tesing code line

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) OAOMalloc(sizeof(int)*edge_list_size);
	
	OAODataTrans( h_graph_edges, StConstrTarget.init(7, 3) );
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	int* h_cost = (int*) OAOMalloc( sizeof(int)*no_of_nodes);
	
	OAODataTrans( h_cost, StConstrTarget.init(7, 3) );
	for(int i=0;i<no_of_nodes;i++)
        h_cost[i]=-1;
        OAOStTrans( h_cost, StConstrTarget.init(3, 3) );
        
	
	OAODataTrans( h_cost, StConstrTarget.init(7, 3) );
	h_cost[source]=0;
	
	printf("Start traversing the tree\n");
	
	int k=0;
#ifdef OPEN
        double start_time = omp_get_wtime();
#ifdef OMP_OFFLOAD
#pragma omp target data map(to: no_of_nodes, h_graph_mask[0:no_of_nodes], h_graph_nodes[0:no_of_nodes], h_graph_edges[0:edge_list_size], h_graph_visited[0:no_of_nodes], h_updating_graph_mask[0:no_of_nodes]) map(h_cost[0:no_of_nodes])
        {
#endif 
#endif
	bool stop;
	do
        {
            //if no thread changes this value then the loop stops
            stop=false;

#ifdef OPEN
            //omp_set_num_threads(num_omp_threads);
    #ifdef OMP_OFFLOAD
    #pragma omp target
    #endif
    OAODataTrans( h_graph_mask, StConstrTarget.init(7, 5) );
    OAODataTrans( h_updating_graph_mask, StConstrTarget.init(7, 5) );
    OAODataTrans( h_graph_visited, StConstrTarget.init(7, 5) );
    OAODataTrans( h_graph_edges, StConstrTarget.init(7, 5) );
    OAODataTrans( h_cost, StConstrTarget.init(7, 5) );
    
    #pragma omp target teams distribute parallel for 
#endif 
            for(int tid = 0; tid < no_of_nodes; tid++ )
            {
                if (h_graph_mask[tid] == true){ 
                    h_graph_mask[tid]=false;
                    for(int i=h_graph_nodes__starting; i<(h_graph_nodes__no_of_edges + h_graph_nodes__starting); i++)
                    {
                        int id = h_graph_edges[i];
                        if(!h_graph_visited[id])
                        {
                            h_cost[id]=h_cost[tid]+1;
                            h_updating_graph_mask[id]=true;
                        }
                    }
                }
            }
            OAOStTrans( h_graph_mask, StConstrTarget.init(5, 4) );
            OAOStTrans( h_updating_graph_mask, StConstrTarget.init(5, 4) );
            OAOStTrans( h_cost, StConstrTarget.init(5, 4) );
            

#ifdef OPEN
    #ifdef OMP_OFFLOAD
    #pragma omp target map(stop)
    #endif
    OAODataTrans( h_graph_mask, StConstrTarget.init(7, 5) );
    OAODataTrans( h_updating_graph_mask, StConstrTarget.init(7, 5) );
    
    #pragma omp target teams distribute parallel for 
#endif
            for(int tid=0; tid< no_of_nodes ; tid++ )
            {
                if (h_updating_graph_mask[tid] == true){
                    h_graph_mask[tid]=true;
                    h_graph_visited[tid]=true;
                    stop=true;
                    h_updating_graph_mask[tid]=false;
                }
            }
            OAOStTrans( h_graph_visited, StConstrTarget.init(5, 4) );
            
            k++;
        }
	while(stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
        printf("Compute time: %lf\n", (end_time - start_time));
#ifdef OMP_OFFLOAD
        }
#endif
#endif
	//Store the result into a file
#ifdef OFFLOADING
    FILE *fpo = fopen("result_out.txt","w");
#else
    FILE *fpo = fopen("result.txt","w");
#endif
	
	OAODataTrans( h_cost, StConstrTarget.init(7, 3) );
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
#ifdef OFFLOADING
    printf("Result stored in result_out.txt\n");
#else
    printf("Result stored in result.txt\n");
#endif


	// cleanup memory
	OAOFree( h_graph_nodes);
	
	OAODataTrans( h_graph_edges, StConstrTarget.init(2, 2) );
	OAOFree( h_graph_edges);
	
	OAODataTrans( h_graph_mask, StConstrTarget.init(2, 2) );
	OAOFree( h_graph_mask);
	
	OAODataTrans( h_updating_graph_mask, StConstrTarget.init(2, 2) );
	OAOFree( h_updating_graph_mask);
	
	OAODataTrans( h_graph_visited, StConstrTarget.init(2, 2) );
	OAOFree( h_graph_visited);
	
	OAODataTrans( h_cost, StConstrTarget.init(2, 2) );
	OAOFree( h_cost);


OAODataTrans( h_graph_mask, StConstrTarget.init(2, 2) );
OAODataTrans( h_updating_graph_mask, StConstrTarget.init(2, 2) );
OAODataTrans( h_graph_visited, StConstrTarget.init(2, 2) );
OAODataTrans( h_graph_edges, StConstrTarget.init(2, 2) );
OAODataTrans( h_cost, StConstrTarget.init(2, 2) );
}

