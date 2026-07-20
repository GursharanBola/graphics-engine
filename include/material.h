#ifndef MATERIAL_H
#define MATERIAL_H
#include <Eigen/Dense>

struct material {
  public:
    // dielectric_const of plastics are ~0.4
    material() : material{0, 0, Eigen::Vector3d{.4, .4, .4}, false} {};
    material(const double shine, const double metalic,
             const Eigen::Vector3d reflectance, const bool is_metal)
        : shine(shine), metalic(metalic), reflectance(reflectance),
          is_metal(is_metal) {}
    double shine;
    double metalic;              // bounded between [0,1]
    Eigen::Vector3d reflectance; // each element bounded between [0,1]
    bool is_metal;               // tells us if the material is a metal
    int metal_data = -1;         // index to its cube mapped reflection buffers
};

#endif
