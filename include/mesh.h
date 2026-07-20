#ifndef MESH_H
#define MESH_H

#include "buffer.h"
#include "material.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

typedef struct triangle {
    // note all of these are v_id.
    int point1;
    int point2;
    int point3;
} triangle;

// the program expects all triangles to have a winding order of CCW, this is
// VERY important allows for backface culling

class mesh {
  public:
    std::vector<triangle> list_of_triangles;

    mesh(const int mesh_id, const color mesh_color,
         std::unique_ptr<material> mat = nullptr,
         const Eigen::Vector3d &origin = Eigen::Vector3d::Zero(),
         const int num_samples = 1)
        : mat(std::move(mat)), mesh_id(mesh_id), num_samples(num_samples),
          mesh_color(mesh_color), origin(origin) {}

    virtual ~mesh() = default;
    virtual void build(vertex_buffer &v_buffer) = 0;
    virtual Eigen::Vector3d find_normal(const Eigen::Vector3d &point) const = 0;

    Eigen::Vector3d get_origin() const { return origin; }
    int get_id() const { return mesh_id; }
    int get_samples() const { return num_samples; }
    color get_color() const { return mesh_color; }

  protected:
    std::unique_ptr<material> mat;
    int mesh_id;
    int num_samples;
    color mesh_color;
    Eigen::Vector3d origin;
    friend class engine;
};

// much like RTIOW, main test object
class sphere : public mesh {
  public:
    sphere(const int mesh_id, const Eigen::Vector3d &center,
           const double radius, const color &mesh_color,
           std::unique_ptr<material> mat, const int num_samples,
           vertex_buffer &v_buff)
        : mesh(mesh_id, mesh_color, std::move(mat), center, num_samples),
          radius(radius) {
        build(v_buff);
    }

    virtual void build(vertex_buffer &v_buffer) override final;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d &point) const override final;

    double get_radius() const { return radius; }

  private:
    double radius;
};

// quads are useful for making backgrounds
class quad : public mesh {
  public:
    quad(const int mesh_id, const Eigen::Vector3d &origin,
         const Eigen::Vector3d &u, const Eigen::Vector3d &v,
         const color &mesh_color, std::unique_ptr<material> mat,
         vertex_buffer &v_buff)
        : mesh(mesh_id, mesh_color, std::move(mat), origin), u(u), v(v) {
        build(v_buff);
    }

    virtual void build(vertex_buffer &v_buffer) override final;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d &point) const override final;

    Eigen::Vector3d get_u() const { return u; }
    Eigen::Vector3d get_v() const { return v; }

  private:
    Eigen::Vector3d u;
    Eigen::Vector3d v;
};

#endif
