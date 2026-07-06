#ifndef PROJECTOR_H
#define PROJECTOR_H

#include "buffer.h"
#include <Eigen/Dense>

// projector acts a utility for lights and cameras
class projector {
  public:
    projector(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
              const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
              const double focal_dist)
        : origin(origin), cam_u(cam_u), cam_v(cam_v), cam_w(cam_w),
          focal_dist(focal_dist) {};

    // get and set functions
    Eigen::Vector3d get_u() const { return cam_u; }
    Eigen::Vector3d get_v() const { return cam_v; }
    Eigen::Vector3d get_w() const { return cam_w; }
    Eigen::Vector3d get_o() const { return origin; }
    double get_f_dist() const { return focal_dist; }

  private:
    Eigen::Vector3d cam_u;
    Eigen::Vector3d cam_v;
    Eigen::Vector3d cam_w;
    Eigen::Vector3d origin;
    double focal_dist;
};

class light : public projector {
  public:
    light(const Eigen::Vector3d &origin, const Eigen::Vector3d &cam_u,
          const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
          const color &in_color, const double focal_dist)
        : projector(origin, cam_u, cam_v, cam_w, focal_dist),
          l_color(in_color) {}
    color get_color() const { return l_color; }
    // TODO: re-determine what functions these projectors should have
  private:
    color l_color;
};

// cameras will not be in a scene and instead will be outside for reasons
// related to games and user experience
class camera : public projector {
    camera(const Eigen::Vector3d origin, const Eigen::Vector3d cam_u,
           const Eigen::Vector3d cam_v, const Eigen::Vector3d cam_w,
           const double focal_dist)
        : projector(origin, cam_u, cam_v, cam_w, focal_dist) {};
    // TODO: re-determine what functions these projectors should have
};

#endif
