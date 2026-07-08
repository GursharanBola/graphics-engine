#ifndef MATERIAL_H
#define MATERIAL_H
#include <Eigen/Dense>

struct material {
  public:
    Eigen::Vector3d val;
    material() : val(0.0, 0.0, 0.0) {}
    material(const double k_a, const double k_d, const double k_s,
             const double shine)
        : val(k_a, k_d, k_s), shine(shine) {}
    double k_a() const { return val[0]; }
    double k_d() const { return val[1]; }
    double k_s() const { return val[2]; }
    void clamp() { val = val.cwiseMin(1.0).cwiseMax(0.0); }
    double shine;
};

#endif
