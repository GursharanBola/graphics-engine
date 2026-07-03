#ifndef MESH_H
#define MESH_H

#include "buffer.h"
#include "material.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

// TODO: there is a tri_ref object and triangle which are the same
// make this a tri_ref
typedef struct triangle {
    // note all of these are v_id.
    int point1;
    int point2;
    int point3;
} triangle;

class mesh {
  public:
    std::vector<triangle> list_of_triangles;

    mesh(const int mesh_id,
         const Eigen::Vector3d color = Eigen::Vector3d::Zero(),
         const std::shared_ptr<material> mat = nullptr,
         const int num_samples = 1)
        : mesh_id(mesh_id), num_samples(num_samples), color(color), mat(mat) {}

    virtual ~mesh() = default;
    virtual void build(vertex_buffer &v_buffer) = 0;
    virtual Eigen::Vector3d find_normal(const Eigen::Vector3d point) const = 0;

    inline int get_id() const { return mesh_id; }
    inline int get_samples() const { return num_samples; }
    inline Eigen::Vector3d get_color() const { return color; }

  protected:
    std::shared_ptr<material> mat;
    int mesh_id;
    int num_samples;
    Eigen::Vector3d color;
};

class sphere : public mesh {
  public:
    sphere(const int mesh_id, const Eigen::Vector3d center, const double radius,
           const Eigen::Vector3d color, const std::shared_ptr<material> mat,
           const int num_samples)
        : mesh(mesh_id, color, mat, num_samples), center(center),
          radius(radius) {}

    virtual void build(vertex_buffer &v_buffer) override;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d point) const override;

    inline Eigen::Vector3d get_center() const { return center; }
    inline double get_radius() const { return radius; }

  private:
    Eigen::Vector3d center;
    double radius;
};

class quad : public mesh {
  public:
    quad(const int mesh_id, const Eigen::Vector3d origin,
         const Eigen::Vector3d u, const Eigen::Vector3d v,
         const Eigen::Vector3d color, const std::shared_ptr<material> mat)
        : mesh(mesh_id, color, mat), origin(origin), u(u), v(v) {};

    virtual void build(vertex_buffer &v_buffer) override;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d point) const override;

    inline Eigen::Vector3d get_u() { return u; }
    inline Eigen::Vector3d get_v() { return v; }

  private:
    Eigen::Vector3d u;
    Eigen::Vector3d v;
    Eigen::Vector3d origin;
};

#endif
