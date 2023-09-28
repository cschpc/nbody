/*****************************************************************************
MIT License

Copyright (c) 2023 CSC HPC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

/* Simple MPI and OpenMP parallel N-body simulation

   Brute-force N^2 algorithm, MPI parallelization with nearest neighbour chain
   Time integration with leap frog method
*/


#include <omp.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <mpi.h>

template<typename T>
struct Bodies
{
  
  std::vector<T> my_masses;
  std::vector<T> others_masses;
  std::vector<T> my_positions_x;
  std::vector<T> my_positions_y;
  std::vector<T> my_positions_z;
  std::vector<T> others_positions_x;
  std::vector<T> others_positions_y;
  std::vector<T> others_positions_z;
  std::vector<T> velocities_x;
  std::vector<T> velocities_y;
  std::vector<T> velocities_z;
  std::vector<T> accelerations_x;
  std::vector<T> accelerations_y;
  std::vector<T> accelerations_z;
    
  size_t nbodies;

  // Default constructor
  Bodies() = default;

  // Allocate at the time of constuction
  Bodies(int n) : nbodies(n) {
    my_masses.resize(n);
    others_masses.resize(n);
    my_positions_x.resize(n);
    my_positions_y.resize(n);
    my_positions_z.resize(n);
    // positions of neighbouring MPI task
    others_positions_x.resize(n);  
    others_positions_y.resize(n);
    others_positions_z.resize(n);
    velocities_x.resize(n);
    velocities_y.resize(n);
    velocities_z.resize(n);
    accelerations_x.resize(n);
    accelerations_y.resize(n);
    accelerations_z.resize(n);
  };

};

template <class T>
void initialize(Bodies<T> &bodies)
{

  // in order to better compare different parallelizations
  // rank 0 initializes all random data
  int ntasks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t nbodies = bodies.nbodies;
  std::vector<T> all_masses;
  std::vector<T> all_positions_x;
  std::vector<T> all_positions_y;
  std::vector<T> all_positions_z;
  std::vector<T> all_velocities_x;
  std::vector<T> all_velocities_y;
  std::vector<T> all_velocities_z;

  if (0 == rank) 
    {
      all_masses.resize(nbodies * ntasks);
      all_positions_x.resize(nbodies * ntasks);
      all_positions_y.resize(nbodies * ntasks);
      all_positions_z.resize(nbodies * ntasks);
      all_velocities_x.resize(nbodies * ntasks);
      all_velocities_y.resize(nbodies * ntasks);
      all_velocities_z.resize(nbodies * ntasks);
  
      std::mt19937 engine(4);

      // For masses let's use normal distribution 
      std::normal_distribution<> mass_dist(20, 8);

      // For positions and velocity uniform distribution
      std::uniform_real_distribution<T> pos_dist(-1000, 1000);
      std::uniform_real_distribution<T> vel_dist(-100, 100);  

      for (size_t i = 0; i < nbodies * ntasks; i++)
        {
          all_masses[i] = mass_dist(engine);
          all_positions_x[i] = pos_dist(engine);
          all_positions_y[i] = pos_dist(engine);
          all_positions_z[i] = pos_dist(engine);
          all_velocities_x[i] = vel_dist(engine);
          all_velocities_y[i] = vel_dist(engine);
          all_velocities_z[i] = vel_dist(engine);
        }

    }

      MPI_Scatter(all_masses.data(), nbodies, MPI_DOUBLE, 
                  bodies.my_masses.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_positions_x.data(), nbodies, MPI_DOUBLE, 
                  bodies.my_positions_x.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_positions_y.data(), nbodies, MPI_DOUBLE, 
                  bodies.my_positions_y.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_positions_z.data(), nbodies, MPI_DOUBLE, 
                  bodies.my_positions_z.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_velocities_x.data(), nbodies, MPI_DOUBLE, 
                  bodies.velocities_x.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_velocities_y.data(), nbodies, MPI_DOUBLE, 
                  bodies.velocities_y.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);
      MPI_Scatter(all_velocities_z.data(), nbodies, MPI_DOUBLE, 
                  bodies.velocities_z.data(), nbodies, MPI_DOUBLE,
                  0, MPI_COMM_WORLD);

      // everybody initializes its own accelerations
      for (size_t i = 0; i < bodies.nbodies; i++)
        {
          bodies.accelerations_x[i] = 0.0;
          bodies.accelerations_y[i] = 0.0;
          bodies.accelerations_z[i] = 0.0;
        }

}

template <class T>
T energy(Bodies<T> &bodies) 
{
  T kinetic_energy = 0.0;
  T potential_energy = 0.0;

  // Kinetic energy is local, so everybody calculates its own part
  for (size_t i=0; i < bodies.nbodies; i++)
    {
      // kinetic energy
      kinetic_energy += 0.5 * bodies.my_masses[i] * ( bodies.velocities_x[i] * bodies.velocities_x[i] +
                                              bodies.velocities_y[i] * bodies.velocities_y[i] +
                                              bodies.velocities_z[i] * bodies.velocities_z[i] );     
    }

  MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // quick and dirty parallelization: rank 0 gathers all 
  // positions and calculates potential energy

  int ntasks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t nbodies = bodies.nbodies;
  size_t all_nbodies = bodies.nbodies * ntasks;
  std::vector<T> all_masses;
  std::vector<T> all_positions_x;
  std::vector<T> all_positions_y;
  std::vector<T> all_positions_z;

  if (0 == rank) 
    {
      all_masses.resize(nbodies * ntasks);
      all_positions_x.resize(nbodies * ntasks);
      all_positions_y.resize(nbodies * ntasks);
      all_positions_z.resize(nbodies * ntasks);
    }
  MPI_Gather(bodies.my_masses.data(), nbodies, MPI_DOUBLE,
             all_masses.data(), nbodies, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  MPI_Gather(bodies.my_positions_x.data(), nbodies, MPI_DOUBLE,
             all_positions_x.data(), nbodies, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  MPI_Gather(bodies.my_positions_y.data(), nbodies, MPI_DOUBLE,
             all_positions_y.data(), nbodies, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  MPI_Gather(bodies.my_positions_z.data(), nbodies, MPI_DOUBLE,
             all_positions_z.data(), nbodies, MPI_DOUBLE,
             0, MPI_COMM_WORLD);


    // potential energy
    if (0 == rank) {
    for (size_t i=0; i < all_nbodies; i++)
      {
        for (size_t j=i; j < all_nbodies; j++)
          {
            auto delta_x = all_positions_x[i] - all_positions_x[j];
            auto delta_y = all_positions_y[i] - all_positions_y[j];
            auto delta_z = all_positions_z[i] - all_positions_z[j];
            auto distance = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
            potential_energy += -all_masses[i] * all_masses[j] / (distance + 1.0e-6);
            
          }
      }
    }
    MPI_Bcast(&potential_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return (kinetic_energy + potential_energy) / (all_nbodies);
}

template <class T>
T average_velocity(Bodies<T> &bodies) 
{
  T v_ave = 0.0;
  // Everybody calculates its own part
  for (size_t i=0; i < bodies.nbodies; i++)
    {
      
      v_ave += sqrt(bodies.velocities_x[i] * bodies.velocities_x[i] +
                    bodies.velocities_y[i] * bodies.velocities_y[i] +
                    bodies.velocities_z[i] * bodies.velocities_z[i]);     
    }

  MPI_Allreduce(MPI_IN_PLACE, &v_ave, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return v_ave;
  
}

template <class T>
void simulate(Bodies<T> &bodies, int niters)
{
  // Leap-frog integration
  const T dt = 0.2;

  int ntasks;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Neighbouring ranks
  int dst = (rank + 1) % ntasks;
  int src = (rank - 1 + ntasks) % ntasks;
  
  size_t nbodies = bodies.nbodies;
  #pragma omp parallel
  for (int n=0; n < niters; n++)
    {
      #pragma omp for simd
      for (size_t i=0; i < nbodies; i++)
      {
        // kick and step
        bodies.velocities_x[i] += 0.5*dt * bodies.accelerations_x[i];
        bodies.velocities_y[i] += 0.5*dt * bodies.accelerations_y[i];
        bodies.velocities_z[i] += 0.5*dt * bodies.accelerations_z[i];

        bodies.my_positions_x[i] += dt * bodies.velocities_x[i];
        bodies.my_positions_y[i] += dt * bodies.velocities_y[i];
        bodies.my_positions_z[i] += dt * bodies.velocities_z[i];
      }

    #pragma omp single
    {
    bodies.others_positions_x = bodies.my_positions_x;
    bodies.others_positions_y = bodies.my_positions_y;
    bodies.others_positions_z = bodies.my_positions_z;
    }

    // calculate accelerations
    for (int nblock = 0; nblock < ntasks; nblock++)
      {
      #pragma omp for
      for (size_t i=0; i < nbodies; i++)
        {
        #pragma omp simd
        for (size_t j=0; j < nbodies; j++)
          {
            auto delta_x = bodies.my_positions_x[i] - bodies.others_positions_x[j];
            auto delta_y = bodies.my_positions_y[i] - bodies.others_positions_y[j];
            auto delta_z = bodies.my_positions_z[i] - bodies.others_positions_z[j];
            auto distance2 = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z;
            auto s = bodies.my_masses[i] / (distance2 * sqrt(distance2) + 1.0e-6);
            bodies.accelerations_x[i] += s * delta_x;
            bodies.accelerations_y[i] += s * delta_y;
            bodies.accelerations_z[i] += s * delta_z;
          }
        }

        // communicate the other data along the chain
        #pragma omp single
        if (nblock < ntasks - 1)
        {
           MPI_Sendrecv_replace(bodies.others_positions_x.data(), nbodies, MPI_DOUBLE, 
                                dst, 1, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           MPI_Sendrecv_replace(bodies.others_positions_y.data(), nbodies, MPI_DOUBLE, 
                                dst, 1, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           MPI_Sendrecv_replace(bodies.others_positions_z.data(), nbodies, MPI_DOUBLE, 
                                dst, 1, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        }

      // kick
      #pragma omp for
      for (size_t i=0; i < nbodies; i++)
        {
        bodies.velocities_x[i] += 0.5*dt * bodies.accelerations_x[i];
        bodies.velocities_y[i] += 0.5*dt * bodies.accelerations_y[i];
        bodies.velocities_z[i] += 0.5*dt * bodies.accelerations_z[i];
        }
    }
}

int main(int argc, char** argv)
{
  if (argc != 3)
    {
      printf("Usage: nbody [number of bodies] [number of iterations]\n");
      exit(-1);
    }

  size_t nbodies = std::atoi(argv[1]);
  int niters = std::atoi(argv[2]);

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided < MPI_THREAD_SERIALIZED) {
        printf("MPI_THREAD_SERIALZED thread support level required\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int rank, ntasks;
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t my_nbodies = nbodies / ntasks;
  if (my_nbodies * ntasks != nbodies) {
     printf("Cannot divide bodies evenly!!! bodies=%ld MPI tasks=%d\n", nbodies, ntasks);
     MPI_Abort(MPI_COMM_WORLD, -1);
  }

  auto bodies = Bodies<double> (my_nbodies);
  initialize(bodies);

  // warm-up
  simulate(bodies, 5);
  
  auto e = energy(bodies);
  if (0 == rank)
      std::cout << "Energy in the beginning: " << e << std::endl;

  auto t0 = omp_get_wtime();
  simulate(bodies, niters);
  auto t1 = omp_get_wtime();

  e = energy(bodies);
  if (rank == ntasks - 1)
      std::cout << "Energy in the end:       " << e << std::endl;

  // print reference and timing 
  auto v_ave = average_velocity(bodies) / nbodies;
  if (0 == rank) {
      std::cout << "Average velocity: " << v_ave << std::endl;
      std::cout << "Time: " << t1 - t0 << " s" << std::endl;
  } 
  

  MPI_Finalize();
}
