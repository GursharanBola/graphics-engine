#ifndef MATERIAL_H
#define MATERIAL_H
#include "buffer.h"
#include <Eigen/Dense>

struct material {
  public:
    // dielectric_const of plastics are ~0.4
    material()
        : material{0, 0, Eigen::Vector3d{.4, .4, .4}, color{1, 1, 1}, false} {};
    material(const double shine, const double metalic,
             const Eigen::Vector3d reflectance, const color col,
             const bool is_metal)
        : shine(shine), metalic(metalic), reflectance(reflectance), col(col),
          is_metal(is_metal) {}

    double shine;                // power of spectral term
    double metalic;              // bounded between [0,1]
    Eigen::Vector3d reflectance; // each element bounded between [0,1]
    color col;                   // the mesh's color
    bool is_metal;               // tells us if the material is a metal
    int metal_data = -1;         // index to its cube mapped reflection buffers
};

#endif
