#ifndef MATERIAL_H
#define MATERIAL_H
#include <Eigen/Dense>

struct material {
  public:
    // dielectric_const of plastics are ~0.4
    material() : material{0, 0, Eigen::Vector3d{.4, .4, .4}} {};
    material(const double shine, const double metalic,
             const Eigen::Vector3d reflectance)
        : shine(shine), metalic(metalic), reflectance(reflectance) {}
    double shine;
    double metalic;              // bounded between [0,1]
    Eigen::Vector3d reflectance; // each element bounded between [0,1]
};

#endif
