#ifndef PROJECTOR_H
#define PROJECTOR_H

#include "buffer.h"
#include <Eigen/Dense>

// projector acts a utility for lights and cameras
class projector {
  public:
    projector(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
              const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
              const double focal_len)
        : origin(origin), cam_u(cam_u), cam_v(cam_v), cam_w(cam_w),
          focal_len(focal_len) {};

    // get and set functions
    Eigen::Vector3d get_u() const { return cam_u; }
    Eigen::Vector3d get_v() const { return cam_v; }
    Eigen::Vector3d get_w() const { return cam_w; }
    Eigen::Vector3d get_o() const { return origin; }
    double get_f_len() const { return focal_len; }

  private:
    Eigen::Vector3d cam_u;
    Eigen::Vector3d cam_v;
    Eigen::Vector3d cam_w;
    Eigen::Vector3d origin;
    double focal_len;
};

class light : public projector {
  public:
    light(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
          const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
          const color &l_color, const double I_d, const double I_s,
          const double focal_len)
        : projector(origin, cam_u, cam_v, cam_w, focal_len), l_color(l_color),
          I_d(I_d), I_s(I_s) {}
    color get_color() const { return l_color; }
    double get_I_d() const { return I_d; }
    double get_I_s() const { return I_s; }

  private:
    color l_color;
    double I_d;
    double I_s;
};

// cameras will not be in a scene and instead will be outside for reasons
// related to games and user experience
class camera : public projector {
  public:
    camera(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
           const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
           const double focal_len)
        : projector(origin, cam_u, cam_v, cam_w, focal_len) {};
};

#endif
